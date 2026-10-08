# Cross-Vehicle CAN Behavioral Profiles

## 1. Objective

Develop transferable CAN behavior profiles despite differences in IDs, payload layouts, scales, observable signals, and driving conditions. Use CAN-Train-and-Test as an initial candidate; verify vehicle coverage, provenance, metadata, and attack definitions before constructing splits.

**Primary question:** Do behavioral roles and relational motifs improve held-out-vehicle anomaly detection and reduce the normal target data needed for adaptation?

Complete DBC reconstruction is optional. Separate success in signal recovery, semantic identification, representation transfer, and attack detection: none automatically establishes the others.

## 2. Assessment of the “platonic invariant”

The useful intuition is that vehicles share abstract patterns: periodic reporting, modular counters, discrete transitions, redundant measurements, and delayed responses. A reusable vocabulary can capture these without requiring shared CAN IDs.

A fixed universal physical subgraph is too strong:

- Dynamics do not uniquely identify meaning. A smooth field could be speed, temperature, pressure, or a filtered command. Entropy and frequency cannot reliably distinguish intent, actuator, and output slots.
- Shared physics does not imply identical observations. Gearing, controller behavior, load, drivetrain, sampling, and gateway visibility change measured relationships. Engine RPM and wheel speed are not interchangeable outputs.
- Relationships are conditional. Pedal demand need not increase speed during braking, uphill driving, traction limitation, or a gear change. EVs and combustion vehicles expose different intermediates.
- Observability varies. Roles can be absent, duplicated, or multiplexed. Forced one-to-one projection discards useful structure.
- Excessive invariance erases attacks. Normalizing away timing, magnitude, or offsets can remove evidence of flooding, bias injection, or frozen values.

**Recommendation:** use a **context-conditioned behavioral role graph**. Keep “platonic invariant” as an informal motivation; treat stability across vehicles as a hypothesis to test.

### Alternative conceptions

| Conception | Shared object | Value | Limitation |
| --- | --- | --- | --- |
| Behavioral role vocabulary | Soft prototypes: counter, discrete state, smooth measurement | Interpretable first implementation | Semantics remain ambiguous |
| Relational motif vocabulary | Patterns of co-change, agreement, delayed response | Uses relationships to disambiguate individual fields | Needs comparable episodes |
| Hierarchical mechanism model | Shared templates with vehicle-specific parameters | Accommodates drivetrain families | Requires stronger assumptions |
| Predictive residual profiles | Standardized violations of local normal predictors | Practical transfer with limited calibration | Transfers deviations rather than physical meaning |

These can coexist. Start with roles and residuals; introduce relational motifs after demonstrating reliable extraction and dependencies. If semantic alignment fails, useful residual transfer can still succeed.

### Define invariance operationally

For an allowed representation change T, seek r(Tx) approximately equal to r(x). Define T narrowly: consistent CAN-ID relabeling, or equivalent encoding conversion accompanied by its decoder update. Signal-level graphs should be permutation-equivariant; pooled summaries can be invariant.

Use adaptation for units/scales, context for vehicle dynamics, and masks for missing observations. Do not enforce invariance to arbitrary timestamp changes, offsets, or payload permutations: those can change behavior or constitute attacks. Sharing a vocabulary does not require identical latent distributions across vehicles.

## 3. Mining pipeline

### A. Ingestion and audit

Parse in bounded batches and store partitioned Parquet. Preserve vehicle, recording/trip, channel, timestamp, CAN ID, payload length, bytes, and labels where available. Keep IDs for grouping and provenance even if shared weights do not use their numeric values.

Audit timestamp units/resolution, ordering, duplicates, gaps, formats, and recording boundaries. Establish vehicle/trip splits before fitting statistics. Never concatenate trips into an apparently continuous sequence.

**Output:** validated event store and split manifest.

### B. Candidate field recovery

Within each vehicle/message family, estimate bit activity, transition rates, neighboring-bit dependencies, and candidate numeric continuity. Evaluate bounded hypotheses for field boundaries, endianness, and signedness. Test modular increments for counters and selector-conditioned behavior for multiplexing. Checksum identification requires separate validation of a dependency or actual checksum algorithm.

Entropy is a screening feature, not a sufficient boundary detector. A deterministic counter can have high marginal entropy but be predictable from its previous value. Different field types can have the same entropy.

Use CAN-D, READ, and LibreCAN as reference methods. Preserve multiple plausible decodings with confidence and an unresolved fallback; avoid exhaustive combinatorial search.

**Output:** field registry with local IDs, bit spans, decoding hypotheses, multiplexing conditions, confidence, and provenance.

### C. Alignment and fingerprints

Distinguish message arrival rate from value-change rate. Profile timing in event time; resample only for analyses requiring sampled sequences. Preserve observation masks and age since last update. Bound forward filling and avoid interpolation across gaps or categorical transitions.

| Family | Descriptors |
| --- | --- |
| Timing | Update interval, jitter, burstiness, missing updates |
| Dynamics | Autocorrelation, spectrum, increments, smoothness, dwell times |
| State structure | Recurring values, transition entropy, switching rate |
| Encoding | Wraparound, quantization, saturation, bit activity |
| Predictability | Persistence/AR residuals; predictive contribution of neighbors |

Use catch22 as a compact generic baseline for suitable numeric series. Add CAN-specific features and handle constants, short sequences, categorical fields, and undefined features explicitly. Preserve raw magnitude and timing alongside normalized descriptors.

**Output:** fingerprints with sample counts, context coverage, and uncertainty.

### D. Motifs and dependency graphs

Mine single-field and small-group episodes with matrix profiles. Compare selected episodes with constrained DTW when duration varies. Use multiple meaningful durations rather than one arbitrary window size.

Estimate sparse edges using lagged association, conditional prediction, and event-triggered responses. Add PCMCI/PCMCI+ as a later experiment after screening candidate pairs. Store edge type, lag, predictive gain, context, and stability across trips.

Call the result a **predictive dependency graph** unless causal assumptions or interventions justify stronger language. Feedback permits cycles in a variable-level graph; a time-unrolled graph is different. Separate scheduling relations from payload relations. Use time-shift/block surrogates and held-out-trip prediction to assess common-trend and autocorrelation artifacts.

**Output:** vehicle-specific graphs and motif catalog with representative episodes.

### E. Shared roles and partial alignment

Start with soft prototypes: smooth measurement, discrete state, counter, event-associated field, redundant measurement, integrity field, and unknown. Assign physical roles such as intent or actuator only with supporting evidence.

Align fields using fingerprints and relational neighborhoods. Allow unmatched fields, multiple fields per role, and uncertain correspondences. Start with nearest-prototype matching; later investigate feature-aware/partial optimal transport or Gromov–Wasserstein alignment. Statistical similarity alone does not establish semantic identity.

Use decoded subsets, standardized diagnostics, or synchronized GPS/IMU as optional anchors. Distinguish blind, weakly anchored, and DBC-assisted experiments. Frame reuse can bootstrap related vehicles but is narrower than general behavioral transfer.

**Output:** shared vocabulary, soft local mappings, and confidence.

## 4. Temporal GNN–Mamba integration

Keep representation mining separate from encoder choice:

1. A vehicle adapter supplies candidate fields, masks, ages, confidence, and context.
2. A shared temporal encoder models each field history; compare TCN and Mamba under equal information and training budgets.
3. A GNN exchanges information over each vehicle's typed dependency graph.
4. An explicit assignment/pooling module summarizes soft roles and motifs. Retain a local residual path for unmatched or vehicle-specific behavior.
5. Prediction heads estimate changes, masked values, timing, or cross-field responses. Score calibrated residuals, protocol violations, and relationship failures.

Standard Mamba does not automatically make graph edges control state transitions. Implement graph mixing explicitly; graph-conditioned SSM parameters would be an additional research design. Selection gates are not anomaly detectors by themselves.

**Training:** use causal windows from training trips. Start with prediction objectives; add alignment losses only for defensible correspondences. Avoid positive augmentations that remove the events needed for detection.

**Inference:** cache known mappings. For unseen vehicles, specify a normal calibration period or report cold-start uncertainty. Freeze trusted reference statistics and monitor drift. Avoid unrestricted online rerouting or normalization that can absorb attacks. Graph estimation using future observations belongs to offline/calibration work, not zero-latency inference.

## 5. Explanations and evaluation

Explain scores using affected local fields, expected versus observed responses, residuals, timing deviations, and mapping confidence. Validate explanations through controlled perturbations or ablations of the identified relationship.

Gate/parameter inspection is exploratory evidence, not exact causal attribution. MambaLRP is a dedicated attribution research starting point. t-SNE/UMAP can illustrate organization but cannot prove invariance or semantic equivalence.

| Experiment | Purpose |
| --- | --- |
| Frozen model on a held-out vehicle | Strict zero-shot transfer |
| Fixed normal target calibration budgets | Adaptation efficiency |
| Unseen normal operating regimes | False alarms from legitimate novelty |
| Timing and payload/relationship attacks separately | Transfer beyond injection-volume artifacts |
| Consistent identifier relabeling | Independence from arbitrary IDs |
| Missing fields and uncertain decodings | Partial-observability robustness |
| Ablations of fingerprints, motifs, graph, alignment, residual path | Identify the source of improvement |

Define zero-shot strictly: no target-fitted decoders, graphs, scalers, or thresholds. Building these from target logs is unsupervised target adaptation. Fixed per-window calculations available during inference differ from dataset-level target fitting.

Fit all learned preprocessing on permitted training/calibration data only. Prevent repeated recording segments or vehicle identities leaking across splits. Report per-vehicle precision–recall metrics, false alarms/hour, event detection rate, delay, calibration budget, and compute cost. Evaluate extraction and roles separately wherever trusted decoded data exist.

**Success criterion:** profiles improve held-out transfer or calibration efficiency without losing attack sensitivity. If roles are unstable, pursue motif or residual transfer rather than claiming recovered physical semantics.

## 6. Joint temporal and graph motifs

Time-series motifs describe recurring shapes; graph patterns describe recurring relationships; joint motifs describe recurring episodes within a connected signal group. This is a proposed combination of existing tools, not a turnkey library capability.

| Tool | Capability | Use here |
| --- | --- | --- |
| python-igraph | Small structural motif counts via motifs_randesu | Initial structural profiles |
| graph-tool | Motif occurrences and significance against randomized graphs | Locate patterns and test enrichment |
| gSpan implementations | Frequent labeled subgraphs across a graph collection | Mine repeated window/vehicle structures |
| SNAP temporal motifs | Ordered timestamped edges within a duration | Later, when defensible interaction events exist |

A frequent pattern is not necessarily statistically enriched. Compare appropriate null models before calling it a significant network motif. Ordinary topology counts do not automatically include continuous node features, edge weights, or lag labels; use explicit attributed matching or discretization where needed.

Start with sparse dependency graphs, extract connected groups of 2–4 fields, and mine episodes within each group using STUMPY or constrained DTW. Store topology, fingerprints, shapes, relative delays, context, occurrence times, and confidence. Match across vehicles with uncertain roles and node permutations. Separate normalized shape from raw timing/magnitude so attacks remain visible.

Example: A rises sharply, B follows, and C/D rise more slowly while agreeing. The joint pattern includes both the episode and relationships A→B, B→C/D, and C↔D. Its transfer utility is a hypothesis to test against temporal-only and graph-only baselines.

SNAP expects timestamped interaction edges; CAN frames are observations, not explicit pairwise interactions. A conversion rule is required. Consecutive message IDs mainly describe scheduling and should not be interpreted as physical interaction events.

Count support across independent episodes, trips, and vehicles rather than overlapping windows alone. Estimate motif vocabularies using training data. Motif discovery can propose groups for analysis, but does not independently establish their edges or causal direction.

References: [igraph](https://python.igraph.org/en/main/api/igraph.GraphBase.html), [graph-tool motifs](https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.clustering.motifs.html), [motif significance](https://graph-tool.skewed.de/static/docs/stable/autosummary/graph_tool.clustering.motif_significance.html), [Python gSpan](https://github.com/betterenvi/gSpan), [SNAP temporal motifs](https://snap.stanford.edu/temporal-motifs/).

## 7. Discovering richer graph edges

### Preserve the meaning of the existing graph

Consecutive-message edges represent arrival-order transitions. Keep them as a scheduling relation or baseline. Separate the **message event graph** (one node per observed frame) from a **type/field dependency graph** (one node per ID or candidate field with histories). Do not collapse these meanings into one unlabeled adjacency.

If nodes are CAN IDs, model per-ID histories of decoded fields or fallback payload-change features. If nodes are individual frames, use the type/field graph as a template for connecting compatible past observations to current observations. Numeric raw-byte similarity is not equivalent to decoded signal similarity.

### Candidate edge types

| Relation | Discovery method | Interpretation |
| --- | --- | --- |
| Arrival transition | Consecutive messages; time-aware transition counts | Scheduling/traffic order |
| Same-source history | Prior observations of the same ID/field | Temporal continuity |
| Lagged association | Lagged correlation or mutual information | Predictive association candidate |
| Conditional prediction | Target-history predictor versus target-plus-source-history predictor | Additional source information |
| Redundancy | Agreement after an independently fitted transform | Related/duplicated measurement |
| Event response | Repeated source change followed by target response | Conditional response pattern |
| Motif-supported | Consistent co-occurrence/order inside recurring episodes | Candidate relationship with episode support |
| Shared behavioral role | Fingerprint/role similarity across local nodes | Structural similarity, not physical influence |

Use physical time, not frame-count lag: different IDs report at different rates. Retain masks, ages, bus channels, and gaps. Delays include reporting/filtering latency and do not necessarily represent physical response time.

### Offline discovery procedure

1. Fit candidate decoding and aligned histories using normal training trips only. Retain value and timing views; distinguish periodic reporting from actual value changes.
2. Screen candidate field pairs with lagged association, change-event coincidence, or motif co-occurrence over a bounded physical-time horizon. Screen signals in relevant operating contexts and use multiple resolutions when needed.
3. For each source i and target j, compare prediction of j at time t using j's past with prediction using j's past plus i's values at selected positive lags. Measure held-out-trip loss reduction. Add selected other predictors/context to reduce common-driver effects; pairwise gain alone does not establish a direct relation.
4. Use time-shift/block surrogate tests, multiple-testing controls, and stability across trips. Keep sparse edges supported by effect size, not p-values alone. Top-k pruning is a capacity limit, not evidence that every selected edge is real.
5. Optionally apply Tigramite PCMCI/PCMCI+ to screened candidates. Its assumptions and conditional independence test must suit the data; report predictive dependencies when causal sufficiency is unsupported.
6. Save an edge registry: source, target, relation, lag or lag interval in seconds, score, confidence, context, support count, and training provenance. Keep several supported lags if warranted; no single optimal delay is guaranteed.

### Make the GNN actually use the delay

An edge attribute saying “200 ms” does not by itself make the GNN read the source 200 ms ago. Implement one of:

- **History-encoded nodes:** temporal encoders summarize causal histories; lag embeddings help the GNN select relevant information. This is approximate unless the history representation retains the required delay information.
- **Lag-indexed messages:** read source hidden state h_i(t−τ) from a timestamp-indexed buffer, or aggregate states over a learned/specified lag interval. Send it to target j at t with relation, lag, masks, and confidence.
- **Time-expanded event graph:** connect past source events to current targets when their separation matches an allowed lag range. Bound neighbors/time horizons to avoid quadratic growth.

For a temporal GNN–Mamba model, start with history-encoded nodes, then test lag-indexed messages. All buffers and message construction must be causal. A bidirectional dependency template can be estimated offline, but inference must never fetch future events.

### Learned and hybrid alternatives

Graph WaveNet and MTGNN provide precedents for learned adjacency in multivariate forecasting. They do not automatically discover explicit delays or identify causality. Their node-specific learned embeddings also do not transfer directly to unseen IDs; use shared feature/history-based edge scoring or calibrated local adapters for cross-vehicle work.

Prefer an interpretable mined graph plus a sparse learned residual graph. Learn candidate weights on normal prediction tasks, retain relation types, and avoid unrestricted online graph adaptation during suspected attacks. Dynamic edge disappearance can itself be evidence of a relationship failure.

### First experiment

Compare: consecutive-only; consecutive plus same-source history; mined lagged edges; mined lagged edges with explicit delayed messages; and the latter with motif-supported groups. Hold representation, temporal encoder, and compute budget comparable. Evaluate prediction gain, normal false alarms, attack detection/delay, graph stability, and edge-construction cost.

References: [Tigramite](https://github.com/jakobrunge/tigramite), [Graph WaveNet](https://arxiv.org/abs/1906.00121), [MTGNN](https://arxiv.org/abs/2005.11650).

## 8. Implementation sequence and tools

| Priority | Deliverable | Tools / constraints |
| --- | --- | --- |
| 1 | Event store and splits | Polars/Arrow/Parquet; bounded batches |
| 2 | Field registry and fingerprints | CAN decoding baselines, NumPy/SciPy, catch22 |
| 3 | Local and shared-profile baselines | Persistence/AR, clustering, simple classifiers |
| 4 | Motifs and sparse dependencies | STUMPY, constrained DTW, Tigramite |
| 5 | Partial role alignment | Prototypes first; optimal transport later |
| 6 | GNN–Mamba and explanations | PyTorch/PyG; compatible scan implementation; residual attribution |

Polars-to-NumPy is zero-copy only under documented layout/type/null conditions. Measure copies and memory. The original entropy example pooled IDs and used a global denominator; compute a separate distribution per intended group.

Chunked ingestion does not make motif discovery, pairwise testing, or DTW inexpensive. Screen fields/edges, select representative episodes, and benchmark before scaling. STUMPY acceleration depends on routine and hardware. Compare constrained exact DTW with approximations rather than assuming FastDTW is fastest.

**First milestone:** establish profile stability and held-out-vehicle utility before building a universal graph or custom graph-conditioned SSM.

## 9. References and their roles

- [CAN-D](https://arxiv.org/abs/2006.05993), [READ](https://doi.org/10.1109/TIFS.2018.2870826), [LibreCAN](https://github.com/mdp93/LibreCAN_CCS): decoding baselines.
- [CANMatch](https://ion-turcanu.net/publication/buscemi2021canmatch/): frame-reuse transfer.
- [catch22](https://arxiv.org/abs/1901.10200): interpretable temporal descriptors.
- [Matrix profile](https://escholarship.org/uc/item/6sk5c12h), [STUMPY](https://stumpy.readthedocs.io/en/latest/Tutorial_STUMPY_Basics.html): motif/discord discovery.
- [PCMCI+](https://arxiv.org/abs/2003.03685): dependency discovery with explicit causal assumptions.
- [Gromov–Wasserstein matching](https://github.com/HongtengXu/s-gwl): relational alignment precedent; CAN application is proposed.
- [TS2Vec](https://arxiv.org/abs/2106.10466): temporal representation precedent, not CAN invariance evidence.
- [Mamba](https://arxiv.org/abs/2312.00752), [MambaLRP](https://arxiv.org/abs/2406.07592): selective SSMs and attribution research.
- [Locatello et al.](https://proceedings.mlr.press/v97/locatello19a.html): unsupervised disentanglement requires inductive biases. This motivates explicit assumptions; it does not rule out CAN transfer using temporal structure or anchors.
- [Polars interoperability](https://docs.pola.rs/api/python/version/2/reference/dataframe/api/polars.DataFrame.to_numpy.html): zero-copy conditions.

## 10. Audit record

Merged repeated pipeline and technique descriptions. Replaced the library-by-library narrative with an implementation table. Removed conversational offers, malformed headings, broken code, duplicate references, and unsupported superlatives.

Corrected claims that entropy determines boundaries, temporal association proves causality, dependencies must form a DAG, signal shapes uniquely identify semantics, Mamba enforces physics automatically, gates yield exact explanations, embedding clusters prove invariance, and Arrow guarantees instantaneous zero-copy conversion.

Added uncertainty, multiplexing, partial observability, context, calibration, leakage controls, falsifiable tests, alternative goals, and criteria for changing direction.
