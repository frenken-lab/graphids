from __future__ import annotations

import numpy as np

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.preprocessing.capture import Capture, Frame
from can_discovery.relationships.physical.vehicle_models import evaluate_vehicle_model


def _field(arb_id, signed=False):
    return FieldCandidate(arb_id,0,16,'little',signed,'unknown',0.,1.,(),
                          (('unknown',1.),),(),2)


def _capture(seed):
    rng = np.random.default_rng(seed)
    n = 1600
    accel = np.clip(np.cumsum(rng.normal(0,2,n)), -50,50)
    speed = np.zeros(n)
    speed[0] = 10000
    for i in range(1,n):
        speed[i] = speed[i-1]+.05*accel[i-1]
    frames = []
    for i in range(n):
        frames.append(Frame(i*.05,2*i,'100',int(speed[i]).to_bytes(2,'little')))
        frames.append(Frame(i*.05+.001,2*i+1,'200',
                            int(accel[i]).to_bytes(2,'little',signed=True)))
    return Capture(str(seed),str(seed),tuple(frames),0,n*.05,0,0,{}, {})


def test_double_integrator_replicates_on_independent_trip():
    result = evaluate_vehicle_model([_capture(1),_capture(2)],
               (_field('100'),_field('200',True)),'model:double_integrator')
    assert result.status == 'supported'
    assert result.effect > .01
    assert result.valid_samples > 100


def test_dynamic_bicycle_abstains_without_lateral_velocity():
    fields = (_field('100'),_field('200'),_field('300'),_field('400'))
    result = evaluate_vehicle_model([_capture(1)],fields,'model:dynamic_bicycle')
    assert result.status == 'unavailable'
    assert result.reason == 'lateral_velocity_not_observed'


def _turning_capture(seed):
    rng = np.random.default_rng(seed)
    n = 1600
    t = np.arange(n)
    speed = 600 + 100*np.sin(t/130)
    steering = 70*np.sin(t/32) + 20*np.sin(t/11)
    yaw = .003*speed*steering + rng.normal(0,2,n)
    lateral = .01*speed*yaw + rng.normal(0,5,n)
    frames = []
    for i in range(n):
        for j,(arb_id,value,signed) in enumerate((('100',speed,False),
                    ('200',steering,True),('300',yaw,True),('400',lateral,True))):
            frames.append(Frame(i*.05+j*.001,4*i+j,arb_id,
                       int(value[i]).to_bytes(2,'little',signed=signed)))
    return Capture(str(seed),str(seed),tuple(frames),0,n*.05,0,0,{}, {})


def test_kinematic_bicycle_checks_yaw_and_lateral_acceleration():
    trips = [_turning_capture(1),_turning_capture(2)]
    yaw = evaluate_vehicle_model(trips,(_field('100'),_field('200',True),
                                 _field('300',True)),'model:kinematic_yaw')
    lateral = evaluate_vehicle_model(trips,(_field('100'),_field('300',True),
                                     _field('400',True)),'model:kinematic_lateral')
    assert yaw.status == 'supported'
    assert lateral.status == 'supported'
