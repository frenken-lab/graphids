"""DBC signal-name vocabulary used only for assisted fitting and evaluation."""

DBC_NAMES = {
    'accelerator_pedal': frozenset(('GAS_PEDAL', 'GAS_PEDAL_ALT', 'PEDAL_GAS',
        'ACCEL_PEDAL_POS', 'ACCELERATOR_PEDAL', 'THROTTLE_PEDAL')),
    'brake_input': frozenset(('BRAKE_AMOUNT', 'BRAKE_PEDAL', 'BRAKE_PEDAL_POS',
        'BRAKE_PRESSURE', 'BRAKE_PRESSURE1', 'BRAKE_PRESSURE2', 'BRAKE_1', 'BRAKE_2')),
    'throttle_plate': frozenset(('THROTTLE_PLATE_POSITION', 'THROTTLE_POSITION')),
    'engine_rpm': frozenset(('ENGINE_RPM', 'RPM', 'CLUSTER_ENGINE_RPM')),
    'vehicle_speed': frozenset(('SPEED', 'CAR_SPEED', 'CF_CLU_VEHICLESPEED')),
}
