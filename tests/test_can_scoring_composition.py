"""Composable evidence preserves rank math and explicit missingness."""
from types import SimpleNamespace

import pytest

from can_discovery.scoring.composition import rank_evidence, score_record
from can_discovery.scoring.evidence import ScoreEvidence, combine_score
from can_discovery.scoring.physical import physical_factor_evidence
from can_discovery.scoring import physical
from can_discovery.scoring.timing import arrival_timing_evidence
from can_discovery.preprocessing.capture import Frame
from can_discovery.relationships.shared.evidence import FactorEvidence


def test_existing_three_term_score_is_unchanged():
    evidence = rank_evidence(.7, -.04, .09, .6,
                             assisted=True, contrast_available=True)
    result = score_record(evidence)
    assert result['score'] == pytest.approx(.7 - .04 + .09)
    assert [item['name'] for item in result['evidence']] == [
        'unary_payload', 'source_contrast', 'speed_rpm_relation']
    assert all(item['status'] == 'measured' for item in result['evidence'])


def test_unavailable_terms_do_not_change_blind_rank():
    evidence = rank_evidence(.7, 0., 0., None,
                             assisted=False, contrast_available=False)
    assert combine_score(evidence) == .7
    assert [item.status for item in evidence] == ['measured', 'unavailable', 'unavailable']
    with pytest.raises(ValueError, match='duplicate'):
        combine_score((evidence[0], evidence[0]))
    with pytest.raises(ValueError, match='unavailable'):
        ScoreEvidence('bad', 'layout', 'unavailable', .1)
    invalid = rank_evidence(.7, 0., 0., None, assisted=True,
                            contrast_available=True, contrast_valid=False)
    assert invalid[1].status == 'unavailable'


def test_timing_and_physical_adapters_report_measurements_without_votes(monkeypatch):
    empty = SimpleNamespace(sha256='empty', frames=(), known_bus=None)
    assert arrival_timing_evidence([empty], (None, '100', 1)).status == 'unavailable'
    missing = physical_factor_evidence([empty], 'wheel_speed', None)
    assert missing.status == 'unavailable'
    assert missing.reason == 'missing_role_candidate'
    frames = tuple(Frame(float(i) / 10, i, '100', b'\x01') for i in range(40))
    capture = SimpleNamespace(sha256='fit', frames=frames, known_bus=None)
    timing = arrival_timing_evidence([capture], (None, '100', 1))
    assert timing.status == 'measured'
    assert timing.details['period_seconds'][0] == pytest.approx(.1)
    monkeypatch.setattr(physical, 'evaluate_factor', lambda *_:
        FactorEvidence('wheel_speed', ('speed', 'wheel_speed'), 'supported', effect=.4))
    relation = physical_factor_evidence([capture], 'wheel_speed',
                                        (object(), object()))
    assert relation.raw_value == .4
    assert relation.contribution == 0.
    assert relation.capture_hashes == ('fit',)
