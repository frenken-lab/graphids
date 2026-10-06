from __future__ import annotations

import numpy as np

from can_discovery.experiments.picacan import (
    ICE_ADJACENCY,
    ICE_WEIGHTS,
    aligned,
    match_ice,
    outgoing_ranks,
    ratio_dste,
    symbols,
)
from scripts.prepare_picacan_cohort import select_windows


def test_paper_ice_matrices_and_rank_conversion():
    assert tuple(sum(row) for row in ICE_ADJACENCY) == (3, 3, 2, 1, 0)
    assert ICE_WEIGHTS[1][2:] == (-5, -5, -5)
    edges = {(i, j): (abs(ICE_WEIGHTS[i][j]), 1 if ICE_WEIGHTS[i][j] > 0 else -1)
             for i in range(5) for j in range(5) if ICE_ADJACENCY[i][j]}
    assert outgoing_ranks(edges, 5)[0, 4] == 3
    assert match_ice(edges, 5)['status'] == 'unique'
    assert match_ice({}, 4)['status'] == 'incomplete'


def test_double_tie_symbols_and_constant_entropy():
    constant = symbols(np.full(130, 10))
    increasing = symbols(np.arange(130))
    assert len(constant) == 30
    assert constant[0] != increasing[0]
    assert ratio_dste(np.full(130, 1), np.full(130, 2), 1) is None


def test_async_overlap_no_route_join():
    x, y = aligned((np.array([0., .2, .4]), np.array([0, 2, 4])),
                   (np.array([.1, .3, .5]), np.array([1, 3, 5])))
    assert len(x) == 31 and np.allclose(x, y)


def test_longest_disjoint_windows():
    catalog = {'V': [f'{device}/session/{i}/s' for device, count in
             [('a', 21), ('b', 10), ('c', 8)] for i in range(count)]}
    windows = select_windows(catalog, 'V')
    assert list(map(len, windows.values())) == [21, 10, 8]
    assert len({routes[0].split('/')[0] for routes in windows.values()}) == 3
    windows = select_windows(catalog, 'V', {'b/session/0'})
    assert list(map(len, windows.values())) == [21, 9, 8]


def test_truth_metrics_missing_role_is_unassessable():
    from types import SimpleNamespace

    from can_discovery.decoding.signals import FieldCandidate
    from can_discovery.experiments.picacan import Signal
    from scripts.benchmark_picacan import truth_metrics

    signal = Signal('100', 0, 8, 'little', 1)
    field = FieldCandidate('100', 0, 8, 'little', False, 'known', 0., 1., (),
                           (('ENGINE_RPM', 1.),), (), 1)
    frames = [SimpleNamespace(arb_id='100', payload=bytes([i])) for i in range(10)]
    result = truth_metrics([signal], [{'members': [0], 'pivots': [0]}],
                           {'status': 'incomplete', 'matches': []}, [field],
                           [SimpleNamespace(frames=frames)])
    assert result['exact_candidates'] == 1
    assert result['five_role_retrieval'] == 'unassessable_missing_dbc_truth'


def test_baseline_fit_never_opens_target_dbc(monkeypatch):
    from types import SimpleNamespace

    from scripts import benchmark_picacan as runner

    monkeypatch.setattr(runner, 'checked_captures', lambda records: [
        SimpleNamespace(sha256='fitting', frames=[])])
    monkeypatch.setattr(runner, 'bit_flip_fields', lambda captures, max_ids: [])
    monkeypatch.setattr(runner, 'load_dbc_fields', lambda path: (_ for _ in ()).throw(
        AssertionError('target DBC read during fit')))
    result = runner.fit({'vehicle': 'test', 'powertrain': 'ICE', 'dbc': '/missing.dbc',
                         'windows': {'fitting': []}})
    assert result['match']['status'] == 'incomplete'


def test_converted_hash_is_checked_before_loading(tmp_path, monkeypatch):
    from scripts import benchmark_picacan as runner

    path = tmp_path / 'capture.csv'
    path.write_text('tampered')
    monkeypatch.setattr(runner, 'load_capture', lambda path: (_ for _ in ()).throw(
        AssertionError('loaded before hash check')))
    try:
        runner.checked_captures([{'capture': str(path), 'converted_sha256': '0'*64}])
    except ValueError as exc:
        assert 'converted hash changed' in str(exc)
    else:
        raise AssertionError('changed capture accepted')


def test_rank_cost_uses_only_mapped_subgraph():
    edges = {(i, j): (abs(ICE_WEIGHTS[i][j]) * 10,
                      1 if ICE_WEIGHTS[i][j] > 0 else -1)
             for i in range(5) for j in range(5) if ICE_ADJACENCY[i][j]}
    baseline = match_ice(edges, 5)
    # An unrelated sixth node responds sooner than any selected positive edge.
    # It must not change the cost of the five-node candidate.
    edges[0, 5] = (1, 1)
    assert match_ice(edges, 6)['cost'] == baseline['cost']


def test_status_and_counter_fields_are_excluded_without_dbc():
    from types import SimpleNamespace

    from can_discovery.experiments.picacan import Signal, exclude_status_and_counters

    n = 240
    frames = []
    for i in range(n):
        frames.extend((SimpleNamespace(timestamp=i*.01, arb_id='100',
                                       payload=bytes((i % 16,))),
                       SimpleNamespace(timestamp=i*.01, arb_id='101',
                                       payload=bytes((i % 2,))),
                       SimpleNamespace(timestamp=i*.01, arb_id='102',
                                       payload=bytes((int(60+30*np.sin(i/12)),)))))
    fields = [Signal(identifier, 0, 8, 'little', 1) for identifier in ('100', '101', '102')]
    kept, excluded = exclude_status_and_counters(fields, [SimpleNamespace(frames=frames)])
    assert kept == [fields[2]]
    assert {item['reason'] for item in excluded} == {'modular_counter',
                                                      'status_or_insufficient_variation'}


def test_nonunique_five_node_matches_abstain():
    edges = {}
    for offset in (0, 5):
        edges.update({(offset+i, offset+j):
                      (abs(ICE_WEIGHTS[i][j]),
                       1 if ICE_WEIGHTS[i][j] > 0 else -1)
                      for i in range(5) for j in range(5) if ICE_ADJACENCY[i][j]})
    result = match_ice(edges, 10)
    assert result['status'] == 'ambiguous'
    assert len(result['matches']) == 2


def test_ratio_dste_detects_directed_synthetic_delay():
    from can_discovery.experiments.picacan import peak_ratio

    rng = np.random.default_rng(4)
    source = rng.integers(0, 100, 3000).astype(float)
    target = np.roll(source, 7)
    forward = peak_ratio(source, target, range(1, 31))
    reverse = peak_ratio(target, source, range(1, 31))
    assert forward is not None and forward[0] == 7
    assert reverse is not None and forward[0] < reverse[0]


def test_recursive_bfr_byte_order_extraction():
    from can_discovery.experiments.picacan import Signal, _fields_from_rates

    intel = np.r_[np.linspace(.9, .7, 8), np.linspace(.6, .4, 8),
                  np.linspace(.3, .1, 8)]
    motorola = intel.reshape(3, 8)[::-1].ravel()
    assert Signal('100', 0, 24, 'little', 3) in _fields_from_rates('100', 3, intel)
    assert Signal('100', 0, 24, 'big', 3) in _fields_from_rates('100', 3, motorola)


def test_verified_checksum_field_is_excluded():
    from types import SimpleNamespace

    from can_discovery.experiments.picacan import Signal, exclude_status_and_counters

    frames = []
    for i in range(256):
        first = (i * 17) % 256
        second = (i * 71) % 256
        frames.append(SimpleNamespace(timestamp=i*.01, arb_id='100',
                                      payload=bytes((first, second, (first+second+9)%256))))
    candidate = Signal('100', 16, 8, 'little', 3)
    kept, excluded = exclude_status_and_counters([candidate], [SimpleNamespace(frames=frames)])
    assert not kept
    assert excluded[0]['reason'] == 'verified_sum_or_xor_checksum'


def test_delay_split_keeps_retransmission_and_separates_later_response(monkeypatch):
    from can_discovery.experiments import picacan

    rng = np.random.default_rng(21)
    x = rng.normal(size=800)
    traces = {0: x, 1: np.roll(x, 2), 2: np.roll(x, 30)}
    times = np.arange(800)*.01
    monkeypatch.setattr(picacan, 'series', lambda capture, signal: (times, traces[signal]))
    grouped = picacan.groups([0, 1, 2], [object()], split_lag_bins=10)
    assert [item['members'] for item in grouped] == [[0, 1], [2]]
    assert grouped[0]['pivots'] == [0]


def test_topology_cost_accepts_extra_tecg_edge():
    edges = {(i, j): (abs(ICE_WEIGHTS[i][j]),
                      1 if ICE_WEIGHTS[i][j] > 0 else -1)
             for i in range(5) for j in range(5) if ICE_ADJACENCY[i][j]}
    edges[0, 1] = (1, 1)
    result = match_ice(edges, 5)
    assert result['status'] == 'unique'
    assert result['topology_cost'] == 1
    assert result['matches'] == [(0, 1, 2, 3, 4)]


def test_hev_ev_mode_template_is_separate_from_ice():
    from can_discovery.experiments.picacan import (
        HEV_EV_ADJACENCY,
        HEV_EV_EQUIVALENCE,
        HEV_EV_WEIGHTS,
    )

    assert HEV_EV_WEIGHTS[1][2:] == (-3, -3)
    assert HEV_EV_ADJACENCY[2][3] == 0
    assert HEV_EV_EQUIVALENCE == ((2, 3),)
