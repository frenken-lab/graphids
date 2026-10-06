"""Version 1 experimental PicaCAN ICE graph. No production profile imports this module."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from can_discovery.decoding.field_values import extract

ROLES = ('accelerator', 'brake', 'throttle', 'engine_speed', 'vehicle_speed')
ICE_ADJACENCY = ((0, 0, 1, 1, 1), (0, 0, 1, 1, 1),
                 (0, 0, 0, 1, 1), (0, 0, 0, 0, 1), (0, 0, 0, 0, 0))
ICE_WEIGHTS = ((0, 0, 1, 2, 3), (0, 0, -5, -5, -5),
               (0, 0, 0, 1, 2), (0, 0, 0, 0, 1), (0, 0, 0, 0, 0))
HEV_EV_ROLES = ('accelerator', 'brake', 'drive_motor_speed', 'vehicle_speed')
HEV_EV_ADJACENCY = ((0, 0, 1, 1), (0, 0, 1, 1),
                    (0, 0, 0, 0), (0, 0, 0, 0))
HEV_EV_WEIGHTS = ((0, 0, 1, 2), (0, 0, -3, -3),
                  (0, 0, 0, 0), (0, 0, 0, 0))
# Fig. 7's red box is an equivalence constraint, not a causal edge.
HEV_EV_EQUIVALENCE = ((2, 3),)
GRID_SECONDS = .01
PEARSON_LAGS = range(-100, 101)
RATIO_DELAYS = range(1, 301)


@dataclass(frozen=True)
class Signal:
    arb_id: str
    start: int
    width: int
    endian: str
    length: int


def _fields_from_rates(arb_id, length, rates):
    """Paper Fig. 9 byte runs and recursive Intel/Motorola byte joins."""
    runs = []
    for byte in range(length):
        current = []
        for bit in range(8):
            pos = byte * 8 + bit
            if rates[pos] > 0 and (not current or rates[pos] <= rates[current[-1]]):
                current.append(pos)
            else:
                if len(current) >= 2:
                    runs.append((byte, current[0] % 8, len(current)))
                current = [pos] if rates[pos] > 0 else []
        if len(current) >= 2:
            runs.append((byte, current[0] % 8, len(current)))
    leading = {(byte, width) for byte, bit, width in runs if bit == 0}
    multi = []
    owned = set()
    for byte, bit, width in runs:
        if bit != 0 or width != 8:
            continue
        # Intel: low byte first, subsequent high bytes start at bit zero.
        total = 8
        next_byte = byte + 1
        previous = byte
        while next_byte < length:
            options = [w for b, w in leading if b == next_byte]
            if not options or rates[next_byte*8] >= rates[previous*8+7]:
                break
            next_width = max(options)
            total += next_width
            multi.append(Signal(arb_id, byte*8, total, 'little', length))
            owned.update((previous, next_byte))
            if next_width != 8:
                break
            previous, next_byte = next_byte, next_byte + 1
        # Motorola: low byte last, preceding high bytes start at bit zero.
        total = 8
        previous = byte - 1
        lower = byte
        while previous >= 0:
            options = [w for b, w in leading if b == previous]
            if not options or rates[previous*8] >= rates[lower*8+7]:
                break
            previous_width = max(options)
            total += previous_width
            start = previous*8 + (8-previous_width)
            multi.append(Signal(arb_id, start, total, 'big', length))
            owned.update((previous, lower))
            if previous_width != 8:
                break
            lower, previous = previous, previous - 1
    singles = []
    for byte, bit, width in runs:
        if byte not in owned or bit != 0:
            singles.append(Signal(arb_id, byte*8+bit, width, 'little', length))
    return list(dict.fromkeys((*singles, *multi)))


def bit_flip_fields(captures, *, max_ids=None):
    """Extract BFR fields from every observed ID unless an audit cap is declared."""
    counts = Counter(f.arb_id for c in captures for f in c.frames if f.payload)
    ids = counts.most_common(max_ids)
    out = []
    for arb_id, _ in ids:
        streams = defaultdict(list)
        for capture in captures:
            local = defaultdict(list)
            for frame in capture.frames:
                if frame.arb_id == arb_id and frame.payload:
                    local[len(frame.payload)].append(frame.payload)
            for length, values in local.items():
                if len(values) > 2:
                    bits = np.unpackbits(np.frombuffer(b''.join(values), dtype=np.uint8).reshape(-1, length),
                                         axis=1, bitorder='little')
                    streams[length].append((np.count_nonzero(np.diff(bits.astype(np.int8), axis=0),
                                                              axis=0), len(values)-1))
        for length, records in streams.items():
            flips = sum((a for a, _ in records), np.zeros(length*8))
            rates = flips / sum(n for _, n in records)
            out.extend(_fields_from_rates(arb_id, length, rates))
    return list(dict.fromkeys(out))


def _verified_checksum_byte(recordings, byte):
    """Conservative sum/XOR check on independent held-out halves of each route."""
    if not recordings or any(len(payloads) < 80 for payloads in recordings):
        return False
    for method in ('sum', 'xor'):
        offsets = []
        for payloads in recordings:
            sample = np.frombuffer(b''.join(payloads[::max(1, len(payloads)//2048)]),
                                   dtype=np.uint8).reshape(-1, len(payloads[0]))
            if sample.shape[0] < 80 or len(np.unique(sample[:, byte])) < 32:
                break
            other = np.delete(sample, byte, axis=1)
            prediction = (other.astype(np.uint16).sum(axis=1) % 256).astype(np.uint8) \
                if method == 'sum' else np.bitwise_xor.reduce(other, axis=1)
            residual = (sample[:, byte].astype(np.int16)-prediction.astype(np.int16)) % 256 \
                if method == 'sum' else np.bitwise_xor(sample[:, byte], prediction)
            cut = len(residual)//2
            offset = Counter(residual[:cut].tolist()).most_common(1)[0][0]
            if np.mean(residual[cut:] == offset) < .99:
                break
            offsets.append(int(offset))
        if len(offsets) == len(recordings) and len(set(offsets)) == 1:
            return True
    return False


def exclude_status_and_counters(signals, captures):
    """Fit-only status, modular-counter, and verified checksum exclusions."""
    kept, excluded = [], []
    families = {(signal.arb_id, signal.length) for signal in signals}
    payloads = []
    for capture in captures:
        indexed = defaultdict(list)
        for frame in capture.frames:
            if frame.payload and (frame.arb_id, len(frame.payload)) in families:
                indexed[frame.arb_id, len(frame.payload)].append(frame.payload)
        payloads.append(indexed)
    for signal in signals:
        recordings = [series(capture, signal)[1] for capture in captures]
        values = np.concatenate([v for v in recordings if len(v)]) if any(
            len(v) for v in recordings) else np.array([])
        reason = None
        if len(values) < 100 or len(np.unique(values)) <= 4:
            reason = 'status_or_insufficient_variation'
        elif signal.width <= 8:
            increments = [np.diff(v.astype(np.int64)) % (1 << signal.width)
                          for v in recordings if len(v) > 1]
            if increments and all(len(delta) >= 40 and np.mean(delta == 1) >= .9
                                  for delta in increments):
                reason = 'modular_counter'
        if reason is None and signal.width == 8 and signal.start % 8 == 0:
            key = (signal.arb_id, signal.length)
            route_payloads = [index.get(key, []) for index in payloads]
            if _verified_checksum_byte(route_payloads, signal.start//8):
                reason = 'verified_sum_or_xor_checksum'
        if reason:
            excluded.append({'signal': signal, 'reason': reason})
        else:
            kept.append(signal)
    return kept, excluded


def series(capture, signal):
    pairs = [(f.timestamp, extract(f.payload, signal.start, signal.width, signal.endian, False))
             for f in capture.frames if f.arb_id == signal.arb_id and f.payload
             and len(f.payload) == signal.length]
    if not pairs:
        return np.array([]), np.array([])
    t, v = map(np.asarray, zip(*pairs))
    unique = np.r_[True, np.diff(t) > 0]
    return t[unique], v[unique].astype(float)


def aligned(left, right, *, grid=GRID_SECONDS):
    """Interpolate only within each capture's overlap; caller never joins routes."""
    tx, x = left
    ty, y = right
    if min(len(tx), len(ty)) < 2:
        return np.array([]), np.array([])
    start, end = max(tx[0], ty[0]), min(tx[-1], ty[-1])
    if end <= start:
        return np.array([]), np.array([])
    stamps = start + np.arange(int((end-start)/grid)+1)*grid
    return np.interp(stamps, tx, x), np.interp(stamps, ty, y)


def delayed_pearson(x, y, lags=PEARSON_LAGS):
    best = (-np.inf, 0)
    for lag in lags:
        a, b = (x[:-lag], y[lag:]) if lag > 0 else ((x[-lag:], y[:lag]) if lag < 0 else (x, y))
        if len(a) < 20 or np.std(a) == 0 or np.std(b) == 0:
            continue
        value = float(np.corrcoef(a, b)[0, 1])
        if value > best[0]:
            best = value, lag
    return best


def groups(signals, captures, *, threshold=.95, split_lag_bins=10,
           coequal=False):
    """Correlate, split distinct delay scales, and solve earliest pivots.

    The paper does not specify a numerical split threshold. Ten 10 ms bins
    is a declared 100 ms separation between near-synchronous copies and
    distinct delayed responses; results must be reported against this choice.
    """
    traces = [[series(capture, signal) for signal in signals] for capture in captures]
    parents = list(range(len(signals)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    delays = {}
    for i in range(len(signals)):
        for j in range(i+1, len(signals)):
            pairs = [aligned(route[i], route[j]) for route in traces]
            if any(len(a) < 200 for a, _ in pairs):
                continue
            scores = [delayed_pearson(a, b) for a, b in pairs]
            if min(score for score, _ in scores) >= threshold:
                parents[root(j)] = root(i)
                delays[i, j] = (int(np.median([lag for _, lag in scores])),
                                min(score for score, _ in scores))
    components = defaultdict(list)
    for i in range(len(signals)):
        components[root(i)].append(i)
    result = []
    for indices in components.values():
        if len(indices) == 1:
            result.append({'members': indices, 'pivots': indices, 'offset_bins': {indices[0]: 0.}})
            continue
        anchor = indices[0]
        local = {node: pos for pos, node in enumerate(indices)}
        design, observed = [], []
        for (i, j), (lag, score) in delays.items():
            if i in local and j in local:
                row = np.zeros(len(indices)-1)
                if i != anchor:
                    row[local[i]-1] = -score
                if j != anchor:
                    row[local[j]-1] = score
                design.append(row)
                observed.append(score*lag)
        fitted = np.linalg.lstsq(np.stack(design), np.asarray(observed), rcond=None)[0]
        offsets = {anchor: 0.}
        offsets.update({node: float(fitted[pos-1]) for node, pos in local.items() if pos})
        ordered = sorted(indices, key=lambda node: (offsets[node], node))
        clusters = [[ordered[0]]]
        for node in ordered[1:]:
            if offsets[node]-offsets[clusters[-1][-1]] >= split_lag_bins:
                clusters.append([])
            clusters[-1].append(node)
        for members in clusters:
            first = offsets[members[0]]
            pivots = [node for node in members if offsets[node]-first < 1.]
            result.append({'members': sorted(members),
                           'pivots': pivots if coequal else pivots[:1],
                           'offset_bins': {node: offsets[node] for node in members}})
    return result


def symbols(values, *, dimension=6, spacing=20):
    """Both stable tie orders in equation 2, encoded as one categorical state."""
    values = np.asarray(values)
    if len(values) < (dimension-1)*spacing+1:
        return np.empty(0, dtype=np.int32)
    windows = np.column_stack([values[(dimension-1)*spacing-k*spacing:len(values)-k*spacing]
                               for k in range(dimension)])
    forward = np.argsort(windows, axis=1, kind='stable')
    backward = np.argsort(-windows, axis=1, kind='stable')[:, ::-1]
    return np.asarray([sum(int(p)*(dimension**k) for k, p in enumerate(np.r_[a, b]))
                       for a, b in zip(forward, backward)], dtype=np.int64)


def _compact_symbols(sx, sy):
    combined = np.concatenate((sx, sy))
    _, codes = np.unique(combined, return_inverse=True)
    return codes[:len(sx)].astype(np.int64), codes[len(sx):].astype(np.int64), \
        int(codes.max()) + 1


def _entropy(codes):
    counts = np.unique(codes, return_counts=True)[1]
    n = counts.sum()
    return float(np.log(n) - np.sum(counts * np.log(counts)) / n)


def _ratio_dste_compact(sx, sy, radix, delay):
    if len(sx) <= delay or len(sy) <= delay:
        return None
    a, b, future = sx[:-delay], sy[:-delay], sy[delay:]
    by = b * radix + future
    bx = b * radix + a
    joint = bx * radix + future
    base = _entropy(by) - _entropy(b)
    if base <= 1e-12:
        return None
    transfer = _entropy(by) + _entropy(bx) - _entropy(b) - _entropy(joint)
    return max(0., float(transfer / base))


def ratio_dste(x, y, delay):
    """I(Y future; X present | Y present) / H(Y future | Y present)."""
    sx, sy, radix = _compact_symbols(symbols(x), symbols(y))
    return _ratio_dste_compact(sx, sy, radix, delay)


def peak_ratio(x, y, delays=RATIO_DELAYS):
    sx, sy, radix = _compact_symbols(symbols(x), symbols(y))
    curve = [(lag, _ratio_dste_compact(sx, sy, radix, lag)) for lag in delays]
    curve = [(lag, value) for lag, value in curve if value is not None]
    if len(curve) < 3:
        return None
    values = np.array([v for _, v in curve])
    if np.all(np.diff(values) >= -1e-12) or np.all(np.diff(values) <= 1e-12):
        return None
    peak = int(np.argmax(values))
    if peak in (0, len(values)-1):
        return None
    return curve[peak][0], curve[peak][1]


def directed_edge(x, y):
    if min(len(x), len(y)) < 2000 or min(np.count_nonzero(np.diff(x)), np.count_nonzero(np.diff(y))) < 20:
        return None
    xy, yx = peak_ratio(x, y), peak_ratio(y, x)
    if xy is None or yx is None or xy[0] == yx[0]:
        return None
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr) or corr == 0:
        return None
    return (0, 1, xy[0], int(np.sign(corr))) if xy[0] < yx[0] else (1, 0, yx[0], int(np.sign(corr)))


def outgoing_ranks(edges, nodes, selection=None):
    """Rank delays inside the candidate subgraph, as in paper section V-C."""
    ranks = {}
    selected = set(range(nodes)) if selection is None else set(selection)
    for source in selected:
        outgoing = sorted(((abs(delay), target, sign) for (a, target), (delay, sign) in edges.items()
                           if a == source and target in selected))
        for rank, (_, target, sign) in enumerate(outgoing, 1):
            ranks[source, target] = rank if sign > 0 else -rank
    return ranks


def match_ice(edges, nodes, *, brake_sign_only=False):
    """Directed topology J1, then outgoing-rank J2, for the ICE query.

    Required query edges must be observed. Additional edges among selected
    nodes contribute to J1; missing query edges make a five-role retrieval
    incomplete under the benchmark's abstention rule.
    """
    if nodes < 5:
        return {'status': 'incomplete', 'topology_cost': None,
                'rank_cost': None, 'matches': []}
    best_topology, best_rank, matches = None, None, []
    outgoing = {i: {j for (a, j) in edges if a == i} for i in range(nodes)}
    for accelerator in range(nodes):
        for brake in range(nodes):
            if brake == accelerator:
                continue
            common = outgoing[accelerator] & outgoing[brake]
            for throttle in common - {accelerator, brake}:
                for engine in (common & outgoing[throttle]) - {accelerator, brake, throttle}:
                    for speed in (common & outgoing[throttle] & outgoing[engine]) - {
                            accelerator, brake, throttle, engine}:
                        mapping = (accelerator, brake, throttle, engine, speed)
                        topology = sum(
                            int(bool(ICE_ADJACENCY[i][j]) != ((mapping[i], mapping[j]) in edges))
                            for i in range(5) for j in range(5) if i != j)
                        if best_topology is not None and topology > best_topology:
                            continue
                        ranked = outgoing_ranks(edges, nodes, mapping)
                        rank_cost = 0
                        for i in range(5):
                            for j in range(5):
                                if not ICE_ADJACENCY[i][j]:
                                    continue
                                observed = ranked[mapping[i], mapping[j]]
                                expected = ICE_WEIGHTS[i][j]
                                if brake_sign_only and expected == -5:
                                    rank_cost += 0 if observed < 0 else 25
                                else:
                                    rank_cost += (observed-expected)**2
                        if best_topology is None or topology < best_topology or \
                                (topology == best_topology and rank_cost < best_rank):
                            best_topology, best_rank, matches = topology, rank_cost, [mapping]
                        elif topology == best_topology and rank_cost == best_rank:
                            matches.append(mapping)
    return {'status': 'unique' if len(matches) == 1 else
            ('ambiguous' if matches else 'incomplete'),
            'topology_cost': best_topology, 'rank_cost': best_rank,
            'cost': best_rank, 'matches': matches}
