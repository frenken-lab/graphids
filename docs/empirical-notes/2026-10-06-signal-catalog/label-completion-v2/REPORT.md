# ID-first DBC annotation completion, development v2

This is a review ledger against DBC annotations. A proposed name remains a role hypothesis; no DBC was edited and no physical role was independently verified.

The pinned manifest uses two fitting routes, two development routes, and one sealed final route per vehicle. The final routes were not opened for this report. Scores are ranks, not confidence probabilities. The method ranks message families by their best candidate score, details eight families, and retains eight ranked field-layout alternatives per family. Relationship factors were not used.

## Development gate

| Measure | Frozen v1 | v2 |
| --- | ---: | ---: |
| Assessable cases | 9 | 10 |
| ID top one | 1 | 0 |
| ID top five | 7 | 6 |
| ID top eight | — | 7 |
| Exact layout top one | 0 | 1 |
| Exact layout top eight | 0 | 2 |
| Exact layout in enumerated pool | 0 | 7 |
| Held-out raw-value agreement | — | 6 |
| Physical role rank one, conditional on exact retained layout | — | 1 |
| Ambiguous detailed families | — | 157 |
| Abstaining role ledgers | — | 20 |

Final-route assessment permitted: **no**. The gate requires at least 7 ID top-five cases and at least one exact-layout top-eight case. Pool coverage does not satisfy the layout-rank gate.

On the nine cases shared with v1, ID top-five is 6/9. The additional assessable case is Forester accelerator pedal, ranked ninth. The lost shared case is Prius RPM, ranked sixth.

## Per-vehicle comparison

| Vehicle | v1 ID top five / assessable | v2 ID top five / assessable | v2 exact top eight |
| --- | ---: | ---: | ---: |
| HONDA_CIVIC | 2/3 | 2/3 | 2 |
| HYUNDAI_SONATA | 0/1 | 0/1 | 0 |
| SUBARU_FORESTER | 3/3 | 3/4 | 0 |
| TOYOTA_PRIUS | 2/2 | 1/2 | 0 |

## Every hidden-role case

| Vehicle | Hidden role | Assessable | ID rank | Exact layout rank | Role rank | Pool coverage | Raw agreement |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| HONDA_CIVIC | accelerator_pedal | no | — | — | — | no | no |
| HONDA_CIVIC | brake_input | yes | 15 | — | — | no | no |
| HONDA_CIVIC | throttle_plate | no | — | — | — | no | no |
| HONDA_CIVIC | engine_rpm | yes | 4 | 7 | 1 | yes | yes |
| HONDA_CIVIC | vehicle_speed | yes | 2 | 1 | 2 | yes | yes |
| HYUNDAI_SONATA | accelerator_pedal | no | — | — | — | no | no |
| HYUNDAI_SONATA | brake_input | no | — | — | — | no | no |
| HYUNDAI_SONATA | throttle_plate | no | — | — | — | no | no |
| HYUNDAI_SONATA | engine_rpm | no | — | — | — | no | no |
| HYUNDAI_SONATA | vehicle_speed | yes | 23 | — | — | no | no |
| SUBARU_FORESTER | accelerator_pedal | yes | 9 | — | — | no | no |
| SUBARU_FORESTER | brake_input | yes | 3 | — | — | yes | yes |
| SUBARU_FORESTER | throttle_plate | no | — | — | — | no | no |
| SUBARU_FORESTER | engine_rpm | yes | 3 | — | — | yes | yes |
| SUBARU_FORESTER | vehicle_speed | yes | 4 | — | — | yes | yes |
| TOYOTA_PRIUS | accelerator_pedal | no | — | — | — | no | no |
| TOYOTA_PRIUS | brake_input | no | — | — | — | no | no |
| TOYOTA_PRIUS | throttle_plate | no | — | — | — | no | no |
| TOYOTA_PRIUS | engine_rpm | yes | 6 | — | — | yes | no |
| TOYOTA_PRIUS | vehicle_speed | yes | 3 | — | — | yes | yes |

A layout in the exhaustive candidate pool is only coverage. Role rank is conditional on retaining the exact reference layout; unary role ranking is not independent physical verification. Historical v1 artifacts remain in the parent directory.
