from __future__ import annotations

import numpy as np

from can_discovery.decoding.signals import FieldCandidate
from can_discovery.preprocessing.capture import Capture, Frame
from can_discovery.relationships.physical.observational_edges import evaluate_graph_edge


def _field(arb_id):
    return FieldCandidate(arb_id,0,16,'little',False,'unknown',0.,1.,(),
                          (('unknown',1.),),(),2)


def _trip(seed, delay=4):
    rng = np.random.default_rng(seed)
    n = 1600
    driver = np.clip(np.cumsum(rng.normal(0,3,n))+100,0,220)
    response = np.zeros(n)
    response[0] = 15000
    for i in range(1,n):
        response[i] = np.clip(response[i-1]+.05*(driver[max(0,i-delay)]-100)
                              +rng.normal(0,.2),1000,60000)
    frames = []
    for i in range(n):
        for j,(arb_id,value) in enumerate((('100',driver[i]),('200',response[i]))):
            frames.append(Frame(i*.02+j*.001,2*i+j,arb_id,
                                int(value).to_bytes(2,'little')))
    return Capture(str(seed),str(seed),tuple(frames),0,n*.02,0,0,{}, {})


def test_graph_edge_requires_replicated_predictive_gain():
    result = evaluate_graph_edge([_trip(1),_trip(2)],_field('100'),_field('200'),
                                 'throttle_rpm')
    assert result.status == 'supported'
    assert result.gain > result.control_gain
    assert len(result.support) == 2


def test_graph_edge_abstains_without_joint_motion():
    trip = _trip(1)
    missing = _field('300')
    result = evaluate_graph_edge([trip],missing,_field('200'),'throttle_rpm')
    assert result.status == 'unavailable'
