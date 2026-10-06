# PicaCAN Section V implementation crosswalk

The isolated experiment implements the three Section V stages of [Ruan et al. (2025)](../Ruan%20et%20al.%20-%202025%20-%20PicaCAN%20Reverse%20Engineering%20Physical%20Semantics%20of%20Signals%20in%20CAN%20Messages%20Using%20Physically-Induced.pdf) in [the core](../../can_discovery/experiments/picacan.py), [runner](../../scripts/benchmark_picacan.py), and [cohort preparer](../../scripts/prepare_picacan_cohort.py). This is an implementation status report, not a measured reproduction claim. No complete 20/8/8-segment fitting, development, and sealed cohort has been run. The combined chassis graph is separate from the paper baseline.

## A. Physical Signal Extraction

| Paper method | Code and qualification |
| --- | --- |
| Select PT-CAN traces and powertrain-specific SECG | The cohort converts bus 0 and ICE fitting requires a manifest entry with `powertrain: ICE` and a cited evidence source. The catalog platform name alone leaves powertrain `unverified`. Whether bus 0 is PT-CAN for each selected capture still requires capture-level verification. |
| Group frames by CAN ID and calculate bit-flip rates | `bit_flip_fields` computes each bit's adjacent-frame flip rate per ID, payload length, and route; route boundaries are preserved. Every ID is processed by default; `--max-ids` is an explicit audit cap. |
| Find byte-local monotonic BFR runs and single/multi-byte fields | `_fields_from_rates` forms non-increasing runs and recursively joins adjacent runs according to the Intel or Motorola boundary inequality. Zero-flip and singleton runs cannot be identified. Exact behavior for noisy BFR ties and partial bytes is an operational interpretation because the paper gives a diagram rather than executable rules. |
| Decode decimal values with timestamps; exclude nonphysical fields | `series` decodes inferred unsigned values. `exclude_status_and_counters` removes low-cardinality status candidates, modular counters, and sum/XOR checksum bytes verified on held-out portions of every fitting route. Other proprietary checksums and status semantics remain unresolved without labels. Signedness is not inferred in this baseline. |

## B. TECG Generation

| Paper method | Code and qualification |
| --- | --- |
| Interpolate asynchronous traces over shared interval | `aligned` linearly interpolates only within each route overlap on a declared 10 ms grid. |
| Delayed Pearson grouping, lag −100…100, threshold 0.95 | `groups` evaluates each pair on every fitting route; a pair must clear 0.95 on all routes. The 10 ms grid operationalizes the paper's unspecified time unit as ±1 second. |
| Split correlated but temporally distinct semantics and select earliest pivot | Pairwise lag equations are solved jointly. Clusters separated by at least 100 ms are split; the earliest offset is pivot. This numerical split threshold is not specified by the paper. The `coequal_pivots` ablation retains pivots within 10 ms and graph estimation checks all retained combinations. |
| Double ordinal tie ordering, DSTE and RatioDSTE | `symbols` encodes both tie orders with embedding dimension 6 and spacing 20 bins. `ratio_dste` uses empirical categorical entropies equivalent to the paper's conditional-information formula and normalizes by target conditional entropy. Zero entropy is unavailable. |
| Delay peaks, direction, sign, TECG | `peak_ratio` scans 10 ms to 3 seconds, rejects monotonic and endpoint peaks; `directed_edge` assigns the shorter forward/reverse peak delay as direction and Pearson sign as edge sign. Unavailable peaks and disagreeing routes/pivots do not become edges. The finite delay range and activity gates are declared benchmark choices. |

## C. Subgraph Matching

| Paper method | Code and qualification |
| --- | --- |
| Powertrain-specific fixed SECG | ICE five-role adjacency and rank-weight matrices are literal, including brake `−5`. A separate four-role HEV EV-mode template is encoded but not run without identified EV-only intervals. |
| Directed topology cost J1 before rank cost J2 | `match_ice` enumerates mappings containing all nine required ICE edges, counts extra directed edges as J1 mismatches, then minimizes squared signed outgoing-delay rank cost inside each selected five-node subgraph. It preserves equally best mappings. A missing required edge causes an incomplete abstention instead of a forced approximate match. Thus the code intentionally searches a conservative subset of the paper's full nonzero-J1 mapping space. |
| Return physical role assignment | Unique five-node mappings are returned; ambiguous and incomplete matches abstain. The brake `−5` is a literal rank target in the baseline. `brake_sign_only` is a separately declared ablation. |

## Assessment and ablations

The preparer chooses longest unused contiguous windows of at least 20 fitting, 8 development, and 8 sealed segments on separate devices; historical calibration routes are excluded. It pins dataset/OpenPilot/OpenDBC revisions, catalog hash, route/device/session/segment identities, bus, compressed and converted hashes, and DBC hash. It leaves powertrain unverified unless a separate evidence JSON names a source. The runner rechecks converted and DBC hashes.

The runner fits without target DBC access, assesses frozen fields and groups route by route, and only then opens target DBC truth. It reports field coverage for every assessment route, structural and raw-value equivalent boundaries, known semantic-set and pivot labels, directed-edge replication, match ambiguity/abstention, and runtime. Missing DBC truth is explicitly unassessable. The extraction, coequal-pivot, brake-sign, and physical-factor ablations run separately from the baseline. A DBC-oracle **diagnostic** is run after the baseline fit and uses target layouts only in its own extraction; it cannot be cited as blind performance. The optional 10-second shifted-time control is reported separately.

The full benchmark remains **unmeasured** because the long cohort has not been converted and independently verified as ICE PT-CAN. No normal profile has been promoted. The prior expanded-cohort preflight does not count as a sealed PicaCAN result. A complete negative result remains valid; tuning after a sealed assessment needs a new independent final cohort.
