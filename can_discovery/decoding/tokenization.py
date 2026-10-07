"""Capture-local structural CAN tokenization."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from ..preprocessing.traces import MessageTrace


@dataclass(frozen=True, slots=True)
class TokenSpan:
    physical_bits: tuple[int, ...]
    byte_order: str

    @property
    def start(self) -> int:
        return min(self.physical_bits)

    @property
    def width(self) -> int:
        return len(self.physical_bits)


@dataclass(frozen=True, slots=True)
class TokenizationResult:
    trace: MessageTrace
    method: str
    parameters: dict
    spans: tuple[TokenSpan, ...]
    uncovered_bits: tuple[int, ...]
    abstention_reasons: tuple[str, ...] = ()
    alternatives: tuple['LayoutAlternative', ...] = ()


@dataclass(frozen=True, slots=True)
class LayoutAlternative:
    spans: tuple[TokenSpan, ...]
    cost: float
    cut_probabilities: tuple[tuple[int, int, float], ...]
    support: tuple[int, ...]
    provenance: str
    unresolved_bits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FieldLayout:
    start: int
    width: int
    endian: str
    signed: bool


@dataclass(frozen=True, slots=True)
class ExhaustiveTokenizationResult:
    """Lazy alternatives for one field, not a whole-message partition."""
    trace: MessageTrace
    method: str
    max_width: int
    excluded_bits: frozenset[int]

    def candidates(self):
        length = self.trace.payload_length * 8
        seen = set()
        for endian in ('little', 'big'):
            for width in range(1, min(self.max_width, length) + 1):
                for start in range(length - width + 1):
                    bits = (frozenset(range(start, start + width)) if endian == 'little'
                            else frozenset((pos // 8) * 8 + 7 - pos % 8
                                           for pos in range(start, start + width)))
                    if bits & self.excluded_bits:
                        continue
                    for signed in ((False,) if width == 1 else (False, True)):
                        identity = (bits, 'single_bit' if width == 1 else endian, signed)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        yield FieldLayout(start, width, endian, signed)


def _features(trace: MessageTrace):
    arrays = [np.frombuffer(b''.join(sample), dtype=np.uint8).reshape(len(sample), trace.payload_length)
              for sample in trace.samples]
    bits = [np.unpackbits(a, axis=1) for a in arrays]
    flips = np.concatenate([b[1:] != b[:-1] for b in bits if len(b) > 1], axis=0)
    means = np.concatenate(bits).mean(axis=0)
    rates = flips.mean(axis=0) if len(flips) else np.zeros(trace.payload_length * 8)
    return arrays, flips, means, rates


def _partition(cuts, n):
    starts = [0] + [i+1 for i in range(n-1) if cuts[i]]
    ends = [i+1 for i in range(n-1) if cuts[i]] + [n]
    return tuple(TokenSpan(tuple(range(a,b)), 'unresolved') for a,b in zip(starts, ends))


def _can_d_layouts(conditional, rates, length, beta, *, top_k=1, probabilities=None):
    """Enumerate valid byte boundary states and optimize inner cuts."""
    options = [('C', 'B')] + [('C', 'B', 'L') for _ in range(length-2)] + [('C', 'L')]
    if length == 1:
        options = [('C',)]
    ranked = []
    for states in product(*options):
        successor = {}
        predecessor = {}
        valid = True
        for byte, state in enumerate(states):
            if state == 'C':
                continue
            target = byte+1 if state == 'B' else byte-1
            if target in predecessor:
                valid = False
                break
            successor[byte] = target
            predecessor[target] = byte
        if not valid:
            continue
        starts = [byte for byte in range(length) if byte not in predecessor]
        paths = []
        visited = set()
        for start in starts:
            path = []
            byte = start
            while byte not in visited:
                path.append(byte)
                visited.add(byte)
                if byte not in successor:
                    break
                byte = successor[byte]
            paths.append(path)
        if len(visited) != length:
            continue
        cost = 0.
        spans = []
        for byte_path in paths:
            sequence = [bit for byte in byte_path for bit in range(byte*8,byte*8+8)]
            first = 0
            for i in range(len(sequence)-1):
                current, nxt = sequence[i:i+2]
                across = current//8 != nxt//8
                if across and (rates[current] == 0 or rates[nxt] == 0):
                    valid = False
                    break
                if probabilities is None:
                    p = conditional[current,nxt]
                    q = conditional[nxt,sequence[i+2]] if i+2 < len(sequence) else p
                    probability = float(p < .01 or q-p > .5)
                else:
                    probability = float(probabilities[current,nxt])
                mandatory = rates[current] == 0 or rates[nxt] == 0
                cut = not across and (mandatory or beta <= probability)
                if across and mandatory:
                    valid = False
                    break
                cost += beta if cut else probability
                if cut:
                    bits = tuple(sequence[first:i+1])
                    spans.append(TokenSpan(bits, 'big' if len({b//8 for b in bits}) == 1
                                           else ('big' if bits[0] < bits[-1] else 'little')))
                    first = i+1
            if not valid:
                break
            bits = tuple(sequence[first:])
            spans.append(TokenSpan(bits, 'big' if len({b//8 for b in bits}) == 1
                                   else ('big' if bits[0] < bits[-1] else 'little')))
            cost += beta
        if valid:
            spans = tuple(spans)
            cuts = sum(map(lambda s: s.width > 0, spans))
            little = sum(s.byte_order == 'little' for s in spans)
            ranked.append((round(cost, 12), -cuts, little, spans))
    ranked.sort(key=lambda item: item[:3])
    # Nearby inner-cut choices are valid layouts too. Include them before
    # truncating so a one-byte trace can still expose alternatives.
    if top_k > 1:
        expanded = []
        for cost, _, _, spans in ranked[:max(8, top_k)]:
            for index, span in enumerate(spans):
                bits = span.physical_bits
                for split in range(1, len(bits)):
                    a,b = bits[split-1:split+1]
                    if a//8 != b//8:
                        continue
                    p = (float(probabilities[a,b]) if probabilities is not None else
                         float(conditional[a,b] < .01))
                    variant = spans[:index] + (TokenSpan(bits[:split],span.byte_order),
                               TokenSpan(bits[split:],span.byte_order)) + spans[index+1:]
                    expanded.append((round(cost+beta-p,12),-len(variant),
                                     sum(s.byte_order == 'little' for s in variant),variant))
                if index+1 < len(spans):
                    nxt = spans[index+1]
                    a,b = bits[-1],nxt.physical_bits[0]
                    if a//8 == b//8 and span.byte_order == nxt.byte_order and rates[a] and rates[b]:
                        p = (float(probabilities[a,b]) if probabilities is not None else
                             float(conditional[a,b] < .01))
                        variant = spans[:index] + (TokenSpan(bits+nxt.physical_bits,span.byte_order),) + spans[index+2:]
                        expanded.append((round(cost+p-beta,12),-len(variant),
                                         sum(s.byte_order == 'little' for s in variant),variant))
        ranked.extend(expanded)
        ranked.sort(key=lambda item: item[:3])
    unique = []
    seen = set()
    for cost, _, _, spans in ranked:
        key = tuple((s.physical_bits, s.byte_order if s.width > 8 else 'unresolved') for s in spans)
        if key in seen:
            continue
        seen.add(key)
        unique.append((cost, spans))
        if len(unique) == top_k:
            break
    return tuple(unique)


def _can_d_spans(conditional, rates, length, beta):
    return _can_d_layouts(conditional, rates, length, beta)[0][1]


def tokenize(trace: MessageTrace, method: str = 'can_d', *, beta: float = .5,
             bycan_eps: float = .22, bycan_min_samples: int = 2,
             source_model=None, max_layouts: int = 1, max_width: int = 64,
             excluded_bits=()) -> TokenizationResult | ExhaustiveTokenizationResult:
    """Infer a message partition or lazily enumerate unscored field layouts."""
    if method not in ('can_d', 'supervised_can_d', 'bit_flip', 'bycan', 'exhaustive'):
        raise ValueError(f'unknown tokenizer: {method}')
    if method == 'exhaustive':
        if not 1 <= max_width <= 64:
            raise ValueError('max_width must be between 1 and 64')
        return ExhaustiveTokenizationResult(trace, method, max_width,
                                            frozenset(excluded_bits))
    if method == 'supervised_can_d' and source_model is None:
        raise ValueError('supervised_can_d requires an explicit source model')
    if not 1 <= max_layouts <= 8:
        raise ValueError('max_layouts must be between 1 and 8')
    if source_model is not None:
        source_model.check_target(trace.source_hashes)
    n = trace.payload_length * 8
    if min(trace.support) < 2:
        return TokenizationResult(trace, method, {}, (), tuple(range(n)), ('insufficient_frames',))
    arrays, flips, means, rates = _features(trace)
    if method == 'bit_flip':
        cuts = [rates[i] == 0 or rates[i+1] == 0 or rates[i+1] > rates[i]*2 for i in range(n-1)]
        spans = _partition(cuts, n)
        params = {'rate_ratio': 2}
    elif method in ('can_d', 'supervised_can_d'):
        joint = flips.T.astype(float) @ flips.astype(float)
        conditional = joint / np.maximum(flips.sum(axis=0)[:,None], 1)
        probabilities = source_model.cut_probabilities(trace) if method == 'supervised_can_d' else None
        layouts = _can_d_layouts(conditional, rates, trace.payload_length, beta,
                                top_k=max_layouts, probabilities=probabilities)
        spans = layouts[0][1]
        params = {'alpha1': .01, 'alpha2': .5, 'beta': beta, 'max_layouts': max_layouts}
    else:
        from sklearn.cluster import DBSCAN
        whole = np.concatenate(arrays)
        byte_rate = np.concatenate([a[1:] != a[:-1] for a in arrays if len(a)>1]).mean(axis=0)
        byte_features = np.stack((byte_rate, whole.mean(axis=0)/255,
                                  np.array([len(np.unique(whole[:,i]))/256 for i in range(trace.payload_length)])), axis=1)
        labels = DBSCAN(eps=bycan_eps, min_samples=bycan_min_samples).fit_predict(byte_features)
        blocks = []
        i = 0
        while i < trace.payload_length:
            j = i+1
            if j < trace.payload_length and labels[i] >= 0 and labels[i] == labels[j]:
                j += 1
            blocks.append((i*8,j*8))
            i = j
        spans_list = []
        for a,b in blocks:
            lab = DBSCAN(eps=bycan_eps, min_samples=bycan_min_samples).fit_predict(
                np.stack((rates[a:b], means[a:b]),axis=1))
            start = a
            for k in range(a+1,b+1):
                if k == b or lab[k-a] < 0 or lab[k-a] != lab[k-a-1] or lab[k-a-1] < 0:
                    spans_list.append(TokenSpan(tuple(range(start,k)), 'unresolved'))
                    start = k
        spans = tuple(spans_list)
        params = {'eps': bycan_eps, 'min_samples': bycan_min_samples, 'max_bytes': 2}
    active = set(np.flatnonzero(rates > 0))
    spans = tuple(s for s in spans if set(s.physical_bits) & active)
    covered = set().union(*(set(s.physical_bits) for s in spans)) if spans else set()
    alternatives = ()
    if method in ('can_d', 'supervised_can_d'):
        alternatives = tuple(LayoutAlternative(
            tuple(s for s in candidate if set(s.physical_bits) & active), cost,
            tuple((a,a+1,float(probabilities[a,a+1]) if probabilities is not None else
                   float(conditional[a,a+1] < .01))
                  for a in range(n-1) if a//8 == (a+1)//8),
            trace.support, method,
            tuple(sorted(set(range(n)) - set().union(*(set(s.physical_bits) for s in candidate)))))
            for cost,candidate in layouts)
    return TokenizationResult(trace, method, params, spans, tuple(sorted(set(range(n))-covered)),
                              ('all_bits_constant',) if not active else (), alternatives)
