from __future__ import annotations

import csv

import numpy as np
import pytest

from can_discovery.decoding.signals import discover_fields
from can_discovery.preprocessing.capture import audit_capture, load_capture
from can_discovery.preprocessing.tokenization import (
    _can_d_spans,
    prepare_traces,
    tokenize,
)


def _write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('timestamp', 'arbitration_id', 'data_field', 'bus'))
        writer.writerows(rows)


def test_stage_zero_audit_and_independent_bus_length_traces(tmp_path):
    paths = [tmp_path / f'{i}.csv' for i in range(2)]
    for seed,path in enumerate(paths):
        rows = [(float(i), '123', f'{(i+seed)%256:02x}00', 'can0') for i in range(30)]
        rows += [(30., '123', 'xyz', 'can0'), (29., '123', 'abc', 'can0'),
                 (31., '123', '00', 'can1'), (32., '123', '000000', 'can0')]
        _write(path, rows)
    captures = [load_capture(path) for path in paths]
    audit = audit_capture(captures[0])
    assert audit['vehicle'] is None and audit['known_bus'] is None
    assert audit['decreasing_timestamps'] == 1
    assert len(audit['excluded_rows']) == 2
    traces = prepare_traces(captures, samples_per_capture=10)
    assert any(t.bus == 'can0' and t.payload_length == 2 and t.support == (10,10) for t in traces)
    assert all(all(len(p) == t.payload_length for sample in t.samples for p in sample) for t in traces)


def test_methods_share_traces_and_unresolved_order_stays_out_of_numeric_fields(tmp_path):
    paths = [tmp_path / f'{i}.csv' for i in range(2)]
    for seed,path in enumerate(paths):
        _write(path, [(i, '123', (i+seed).to_bytes(2,'little').hex(), 'can0')
                      for i in range(80)])
    captures = [load_capture(path) for path in paths]
    trace, = prepare_traces(captures)
    for method in ('bit_flip','can_d','bycan'):
        result = tokenize(trace,method)
        assert result.method == method and result.trace is trace
        bits = [b for s in result.spans for b in s.physical_bits]
        assert len(bits) == len(set(bits))
        assert set(bits).isdisjoint(result.uncovered_bits)
    registry = discover_fields(captures,method='bycan')
    assert all(f.endian in ('big','little') for f in registry.candidates)
    assert all(s.byte_order == 'unresolved' for r in registry.tokenizations for s in r.spans)
    assert all(f.role in ('counter','checksum') for f in registry.candidates)


def test_version_safety():
    from can_discovery.decoding.field_values import extract
    with pytest.raises(ValueError, match='resolved'):
        extract(b'\x01\x02', 0, 16, 'unresolved', False)
    with pytest.raises(ValueError, match='resolved'):
        extract(b'\x01\x02', 0, 16, 'little', None)


@pytest.mark.parametrize(('bad_edge','expected'), (( (7,8), 'little'), ((15,0), 'big')))
def test_can_d_cross_byte_order_optimizer(bad_edge, expected):
    conditional = np.full((16,16), .5)
    conditional[bad_edge] = 0
    spans = _can_d_spans(conditional, np.full(16,.2), 2, .5)
    assert any(span.width == 16 and span.byte_order == expected for span in spans)


def test_can_d_optimizer_matches_exhaustive_one_byte():
    from itertools import product
    import numpy as np
    from can_discovery.preprocessing.tokenization import _can_d_layouts

    beta = .5
    probabilities = np.zeros((8,8))
    gaps = (.2,.8,.5,.1,.9,.4,.5)
    for bit,p in enumerate(gaps):
        probabilities[bit,bit+1] = p
    layouts = _can_d_layouts(np.zeros((8,8)),np.ones(8),1,beta,
                             top_k=8,probabilities=probabilities)
    costs = []
    for cuts in product((False,True),repeat=7):
        cost = sum(beta if cut else gaps[i] for i,cut in enumerate(cuts)) + beta
        costs.append((cost,-(sum(cuts)+1),cuts))
    expected = min(costs,key=lambda item:item[:2])
    selected = layouts[0]
    observed_cuts = {span.physical_bits[-1] for span in selected[1][:-1]}
    assert abs(selected[0]-expected[0]) < 1e-9
    assert tuple(bit in observed_cuts for bit in range(7)) == expected[2]
    assert len(layouts) == 8


def test_supervised_model_rejects_source_capture_and_incompatible_features():
    from dataclasses import replace
    import pytest
    from can_discovery.preprocessing.source_model import SourceBoundaryModel

    model = SourceBoundaryModel(('source',),('hash-a',),('dbc-hash',),{},object())
    with pytest.raises(ValueError,match='target'):
        model.check_target(('hash-a',))
    with pytest.raises(ValueError,match='target'):
        model.check_target((), 'source')
    with pytest.raises(ValueError,match='incompatible'):
        replace(model,feature_version='future').check_target(())


def test_supervised_feature_vector_has_published_fifteen_components():
    import numpy as np
    from can_discovery.preprocessing.source_model import feature_vector

    flips = np.array([[0,1,0],[1,1,0],[0,0,1]],dtype=bool)
    vector = feature_vector(flips,(0,1,2),0)
    assert vector.shape == (15,)
    np.testing.assert_allclose(vector[10:],vector[5:10]-vector[:5])
