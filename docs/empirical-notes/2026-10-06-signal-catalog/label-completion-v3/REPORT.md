# Controlled ID allocation and relational feedback, development v3

The pinned fitting and development routes are unchanged from v2.1. Final routes were not opened. Hidden DBC annotations are used for transition and oracle-ID evaluation after search choices are made. All names remain role hypotheses.

The controller searches eight IDs normally and sixteen when the score gap between ranks 8 and 16 is at most 0.10. Expanded search, relational reranking, and relational reopening use the same detailed-ID budget per case. The speed–RPM change relation is allowed to affect a target only when two other vehicles pass the source fitting and development gate and the target has a visible opposite-role DBC anchor. Gate thresholds were developed on this cohort; this is a development comparison, not an independent transfer estimate.

## Aggregate outcomes

| Variant | Assessable | ID top 5 | ID top 8 | Exact layout top 1 | Exact layout top 8 | Held-out affine raw agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 10 | 6 | 7 | 1 | 2 | 6 |
| expanded_search | 10 | 6 | 7 | 1 | 2 | 6 |
| relation_fixed_pool | 10 | 6 | 7 | 1 | 2 | 6 |
| relation_with_reopening | 10 | 6 | 7 | 1 | 2 | 6 |

## Where reference hypotheses disappear

| Transition | Cases / assessable |
| --- | ---: |
| Correct ID in initial ranked pool | 10/10 |
| Correct ID in baseline detailed search | 7/10 |
| Exact layout eligible for enumeration in baseline detailed IDs | 7/10 |
| Exact layout retained in baseline top eight | 2/10 |
| Exact layout eligible when reference ID is forced into search | 10/10 |
| Exact layout retained in oracle-ID top eight | 3/10 |
| Oracle-ID held-out affine raw agreement | 6/10 |

The oracle-ID check is diagnostic only: the reference ID is supplied after normal search selection. Eligibility for enumeration is coverage, not a successful retained decode.


## Source gate

| Target | Gate | Supporting source vehicles | Target cases with an eligible anchor |
| --- | --- | --- | ---: |
| HONDA_CIVIC | rejected | SUBARU_FORESTER | 0 |
| HYUNDAI_SONATA | eligible | HONDA_CIVIC, SUBARU_FORESTER | 1 |
| SUBARU_FORESTER | rejected | HONDA_CIVIC | 0 |
| TOYOTA_PRIUS | eligible | HONDA_CIVIC, SUBARU_FORESTER | 2 |

## Per-vehicle outcomes

| Vehicle | Variant | ID top 5 / assessable | Exact layout top 8 |
| --- | --- | ---: | ---: |
| HONDA_CIVIC | baseline | 2/3 | 2 |
| HONDA_CIVIC | expanded_search | 2/3 | 2 |
| HONDA_CIVIC | relation_fixed_pool | 2/3 | 2 |
| HONDA_CIVIC | relation_with_reopening | 2/3 | 2 |
| HYUNDAI_SONATA | baseline | 0/1 | 0 |
| HYUNDAI_SONATA | expanded_search | 0/1 | 0 |
| HYUNDAI_SONATA | relation_fixed_pool | 0/1 | 0 |
| HYUNDAI_SONATA | relation_with_reopening | 0/1 | 0 |
| SUBARU_FORESTER | baseline | 3/4 | 0 |
| SUBARU_FORESTER | expanded_search | 3/4 | 0 |
| SUBARU_FORESTER | relation_fixed_pool | 3/4 | 0 |
| SUBARU_FORESTER | relation_with_reopening | 3/4 | 0 |
| TOYOTA_PRIUS | baseline | 1/2 | 0 |
| TOYOTA_PRIUS | expanded_search | 1/2 | 0 |
| TOYOTA_PRIUS | relation_fixed_pool | 1/2 | 0 |
| TOYOTA_PRIUS | relation_with_reopening | 1/2 | 0 |

## Every hidden-role transition

| Vehicle | Role | Assessable | Initial ID rank | In first 8 | Oracle exact layout rank | Expanded ID rank | Fixed relation ID rank | Reopened ID rank | Budget |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| HONDA_CIVIC | accelerator_pedal | no | — | no | — | — | — | — | 16 |
| HONDA_CIVIC | brake_input | yes | 15 | no | — | 15 | 15 | 15 | 16 |
| HONDA_CIVIC | engine_rpm | yes | 4 | yes | 7 | 4 | 4 | 4 | 8 |
| HONDA_CIVIC | throttle_plate | no | — | no | — | — | — | — | 16 |
| HONDA_CIVIC | vehicle_speed | yes | 2 | yes | 1 | 2 | 2 | 2 | 8 |
| HYUNDAI_SONATA | accelerator_pedal | no | — | no | — | — | — | — | 8 |
| HYUNDAI_SONATA | brake_input | no | — | no | — | — | — | — | 16 |
| HYUNDAI_SONATA | engine_rpm | no | — | no | — | — | — | — | 8 |
| HYUNDAI_SONATA | throttle_plate | no | — | no | — | — | — | — | 8 |
| HYUNDAI_SONATA | vehicle_speed | yes | 23 | no | — | — | — | — | 8 |
| SUBARU_FORESTER | accelerator_pedal | yes | 9 | no | — | 9 | 9 | 9 | 16 |
| SUBARU_FORESTER | brake_input | yes | 3 | yes | 4 | 3 | 3 | 3 | 16 |
| SUBARU_FORESTER | engine_rpm | yes | 3 | yes | — | 3 | 3 | 3 | 8 |
| SUBARU_FORESTER | throttle_plate | no | — | no | — | — | — | — | 16 |
| SUBARU_FORESTER | vehicle_speed | yes | 4 | yes | — | 4 | 4 | 4 | 8 |
| TOYOTA_PRIUS | accelerator_pedal | no | — | no | — | — | — | — | 16 |
| TOYOTA_PRIUS | brake_input | no | — | no | — | — | — | — | 16 |
| TOYOTA_PRIUS | engine_rpm | yes | 6 | yes | — | 6 | 6 | 6 | 16 |
| TOYOTA_PRIUS | throttle_plate | no | — | no | — | — | — | — | 16 |
| TOYOTA_PRIUS | vehicle_speed | yes | 3 | yes | — | 3 | 3 | 3 | 8 |

Relational variants changed a measured case outcome in **0** of 20 cases. The relation was eligible on 2 assessable target cases. This is a ranking cue based on correlated changes, which can reflect shared driving behavior; it does not prove either physical role.

Prius RPM remained at ID rank six after relational reranking; its exact layout was eligible for enumeration but absent from the retained top eight even with oracle ID. Expanded search reached Forester pedal at ID rank nine and Civic brake at rank fifteen, but neither gained an exact retained layout. Forester brake has an additional annotated ID at rank twenty: oracle-ID extraction retained an exact layout there at rank four, beyond the adaptive sixteen-ID budget. These are development diagnostics, not a tuned promotion rule.

Oracle-ID extraction measures layout recovery with the correct ID supplied only to the diagnostic extractor. It is not a deployed variant. Exact layout and held-out affine raw agreement remain separate outcomes.
