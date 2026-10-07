"""Motion-family priority retains the full candidate population."""
from types import SimpleNamespace

from can_discovery.prepare import PreparedFamily
from can_discovery.preprocessing.capture import Frame
from can_discovery.preprocessing.traces import MessageTrace
from can_discovery.scoring.id_priority import motion_id_priority


def test_repeated_co_motion_prioritizes_ids_without_pruning():
    families = tuple(PreparedFamily(MessageTrace(name, 1, None, ((), ()), ('a', 'b')),
                                    frozenset()) for name in ('100', '101', '102'))
    captures = []
    for route in range(2):
        frames = []
        for t in range(64):
            moving = ((t // 8) * 7) % 256
            unrelated = ((t * 37 + route * 11) ^ (t * 13)) % 256
            for name, value in (('100', moving), ('101', moving), ('102', unrelated)):
                frames.append(Frame(float(t), len(frames), name, bytes([value])))
        captures.append(SimpleNamespace(frames=tuple(frames), start=0., end=63.,
                                        known_bus=None))
    ranked = motion_id_priority(families, captures)
    assert len(ranked) == len(families)
    assert {row['arb_id'] for row in ranked[:2]} == {'100', '101'}
    assert ranked[0]['priority'] > ranked[-1]['priority']


def test_four_coactive_wheel_words_prioritize_one_id():
    families = tuple(PreparedFamily(MessageTrace(name, 8, None, ((), ()), ('a', 'b')),
                                    frozenset()) for name in ('200', '201'))
    captures = []
    for route in range(2):
        frames = []
        for t in range(64):
            speed = (t // 8) * 7
            wheels = b''.join((speed + offset).to_bytes(2, 'little') for offset in range(4))
            noise = bytes(((t * (i * 17 + 7) + route * 11) % 256) for i in range(8))
            for name, payload in (('200', wheels), ('201', noise)):
                frames.append(Frame(float(t), len(frames), name, payload))
        captures.append(SimpleNamespace(frames=tuple(frames), start=0., end=63.,
                                        known_bus=None))
    ranked = motion_id_priority(families, captures)
    assert ranked[0]['arb_id'] == '200'
    assert ranked[0]['within_id_score'] > ranked[1]['within_id_score']
