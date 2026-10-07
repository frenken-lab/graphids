"""Source-vehicle supervised CAN-D boundary model.

The artifact is a Python pickle and must be loaded only from a trusted path.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import pickle

import numpy as np

from ..preprocessing.traces import MessageTrace, prepare_traces
from .tokenization import _features
from .signals import _bit_positions

FEATURE_VERSION = 'can-d-flip-15-v1'


def _local(flips: np.ndarray, a: int, b: int) -> np.ndarray:
    x, y = flips[:, a], flips[:, b]
    count = len(x)
    if not count:
        return np.zeros(5)
    def conditional(numerator, denominator):
        return float(np.count_nonzero(numerator)) / max(1, np.count_nonzero(denominator))
    return np.asarray((np.mean(x), conditional(x & y, y), conditional(x & y, x),
                       conditional(~x & ~y, ~y), conditional(~x & ~y, ~x)), dtype=float)


def feature_vector(flips: np.ndarray, sequence: tuple[int, ...], i: int) -> np.ndarray:
    a, b = sequence[i:i+2]
    c = sequence[i+2] if i+2 < len(sequence) else b
    left = _local(flips, a, b)
    right = _local(flips, b, c)
    return np.concatenate((left, right, right-left))


def _sequences(length: int):
    big = tuple(range(length*8))
    little = tuple(bit for byte in reversed(range(length)) for bit in range(byte*8, byte*8+8))
    return big, little


@dataclass(frozen=True)
class SourceBoundaryModel:
    training_vehicles: tuple[str, ...]
    capture_hashes: tuple[str, ...]
    dbc_hashes: tuple[str, ...]
    validation: dict
    classifier: object
    feature_version: str = FEATURE_VERSION
    version: int = 1

    def check_target(self, hashes, vehicle: str | None = None):
        if self.feature_version != FEATURE_VERSION or self.version != 1:
            raise ValueError('incompatible source boundary model features or version')
        if set(hashes) & set(self.capture_hashes) or (vehicle and vehicle in self.training_vehicles):
            raise ValueError('source model contains target vehicle or capture')

    def cut_probabilities(self, trace: MessageTrace) -> np.ndarray:
        self.check_target(trace.source_hashes)
        _, flips, _, rates = _features(trace)
        n = trace.payload_length*8
        result = np.zeros((n,n), dtype=float)
        vectors, pairs = [], []
        for sequence in _sequences(trace.payload_length):
            for i in range(len(sequence)-1):
                a,b = sequence[i:i+2]
                if rates[a] == 0 or rates[b] == 0:
                    result[a,b] = 1.
                else:
                    vectors.append(feature_vector(flips, sequence, i))
                    pairs.append((a,b))
        if vectors:
            scores = self.classifier.predict_proba(np.asarray(vectors))[:,1]
            for pair,score in zip(pairs,scores):
                result[pair] = score
        return result

    def save(self, path: Path):
        path.write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: Path):
        class _ModelUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'can_discovery.preprocessing.source_model' and name == 'SourceBoundaryModel':
                    return cls
                return super().find_class(module, name)

        with path.open('rb') as handle:
            model = _ModelUnpickler(handle).load()
        if not isinstance(model, cls):
            raise ValueError('invalid source boundary model')
        model.check_target(())
        return model


def _examples(traces, fields):
    from .field_values import extract
    vectors, labels, weights = [], [], []
    by_family = {}
    for field in fields:
        if field.mux_start is None:
            by_family.setdefault((field.arb_id.upper(), field.payload_length), []).append(field)
    for trace in traces:
        _, flips, _, rates = _features(trace)
        candidates = by_family.get((trace.arb_id.upper(),trace.payload_length), ())
        active = []
        for field in candidates:
            values = [extract(payload,field.start,field.width,field.endian,field.signed)
                      for sample in trace.samples for payload in sample]
            if len(set(values)) > 1:
                active.append(field)
        known = {}
        for index,field in enumerate(active):
            for bit in _bit_positions(field.start,field.width,field.endian):
                known.setdefault(bit,set()).add(index)
        for sequence in _sequences(trace.payload_length):
            for i in range(len(sequence)-1):
                a,b = sequence[i:i+2]
                if a not in known or b not in known:
                    continue
                if len(known[a]) != 1 or len(known[b]) != 1:
                    continue
                label = int(known[a] != known[b])
                obvious = rates[a] == 0 or rates[b] == 0
                vectors.append(feature_vector(flips,sequence,i))
                labels.append(label)
                weights.append(1 if obvious else (8 if label else 4))
    return np.asarray(vectors), np.asarray(labels), np.asarray(weights)


def train_source_model(vehicles: list[tuple[str, list, Path]], *, samples_per_capture=1000):
    """Each source tuple contains name, three normal captures, and its DBC path."""
    from sklearn.ensemble import RandomForestClassifier
    from .dbc import load_dbc_fields
    from ..preprocessing.traces import prepare_traces
    names, hashes, dbc_hashes = [], [], []
    train_x, train_y, train_w = [], [], []
    validation_sets = []
    for name,captures,dbc in vehicles:
        if len(captures) != 3 or len({c.sha256 for c in captures}) != 3:
            raise ValueError('three independent captures per source vehicle are required')
        names.append(name)
        hashes.extend(c.sha256 for c in captures)
        dbc_hashes.append(sha256(Path(dbc).read_bytes()).hexdigest())
        fields = load_dbc_fields(Path(dbc))
        x,y,w = _examples(prepare_traces(captures[:2],samples_per_capture=samples_per_capture),fields)
        if len(y):
            train_x.append(x); train_y.append(y); train_w.append(w)
        validation_sets.append((name, captures, fields))
    if not train_y or len(set(np.concatenate(train_y))) < 2:
        raise ValueError('source DBC fields provide insufficient boundary labels')
    classifier = RandomForestClassifier(n_estimators=200,max_depth=5,
                 max_features='sqrt',min_samples_leaf=3,random_state=0,n_jobs=-1)
    classifier.fit(np.concatenate(train_x),np.concatenate(train_y),sample_weight=np.concatenate(train_w))
    validation = {}
    for name,captures,fields in validation_sets:
        # The third trip is never used for fitting.
        paired = prepare_traces([captures[0],captures[2]],
                                samples_per_capture=samples_per_capture)
        held_traces = tuple(MessageTrace(t.arb_id,t.payload_length,t.bus,
                            (t.samples[1],),(captures[2].sha256,)) for t in paired)
        x,y,_ = _examples(held_traces,fields)
        if len(y):
            predicted = classifier.predict(x)
            validation[name] = {'examples':len(y),'accuracy':float(np.mean(predicted == y))}
    return SourceBoundaryModel(tuple(names),tuple(hashes),tuple(dbc_hashes),validation,classifier)
