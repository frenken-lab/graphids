"""DBC-to-candidate conversion preserves raw coordinates and mux modes."""
from __future__ import annotations

from types import SimpleNamespace

import cantools

from can_discovery.decoding.dbc import _signal_fields, load_dbc_fields
from can_discovery.decoding.field_values import extract


def test_dbc_layouts_extract_the_same_raw_values_as_cantools(tmp_path):
    path = tmp_path / 'layout.dbc'
    path.write_text('''VERSION ""
NS_ :
BS_:
BU_: ECU
BO_ 291 Example: 8 ECU
 SG_ Little : 0|12@1+ (1,0) [0|4095] "" ECU
 SG_ Big : 31|12@0- (1,0) [-2048|2047] "" ECU
''')
    db = cantools.database.load_file(path)
    payload = db.get_message_by_name('Example').encode({'Little': 1234, 'Big': -321})
    decoded = db.decode_message(291, payload, scaling=False)
    fields = load_dbc_fields(path)
    assert {field.label_hypotheses[0][0]:
            extract(payload, field.start, field.width, field.endian, field.signed)
            for field in fields} == decoded


def test_dbc_signal_with_multiple_mux_values_keeps_every_mode(tmp_path):
    selector = SimpleNamespace(start=7, byte_order='big_endian')
    signal = SimpleNamespace(name='Value', start=8, length=8,
                             byte_order='little_endian', is_signed=False,
                             multiplexer_signal='Mode', multiplexer_ids=[1, 3])
    message = SimpleNamespace(frame_id=291, length=8, name='Example',
                              get_signal_by_name=lambda name: selector)
    fields = _signal_fields(message, signal, tmp_path / 'layout.dbc')
    assert [(field.mux_start, field.mux_value) for field in fields] == [(0, 1), (0, 3)]
