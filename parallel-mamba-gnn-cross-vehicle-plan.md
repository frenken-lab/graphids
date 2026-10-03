# Audited plan: parallel temporal modeling across vehicles

## Goal and scope

The immediate goal is to shorten **end-to-end training of the current
cross-vehicle detector**. Preserve the completed 256-lane pilot and the
running 1,024-lane v2 pilot as fixed baselines. A vehicle graph and learned
reference context are later research extensions. A custom affine scan should
be labeled `ssm_lite_scan`, not Mamba.

Success has two independent requirements: (1) faster end-to-end training at
comparable detection quality; (2) transfer to held-out vehicles using
explicitly declared target context. A scan kernel alone does not prove either.

## Audit findings: the serial dependency is larger than the SSM

The cross-vehicle configuration currently uses an event-level supervised
model with one `ssm_lite` layer and enabled TGN-style per-ID memory. It uses
1,024 segmented training lanes with 32-event chunks; source validation and
target evaluation stay continuous. CUDA graphs are disabled after a
behavioral fit exhausted a 16 GiB V100. Neither rhythm/motif context nor a
GNN is active in these fits. The source split targets 10% of events by whole
recording, and the held-out vehicle's calibration and tests remain separate.

In `HybridScanMixin._forward_lane_with_state`, each event reads its
source/destination ID memory, adds that context to the encoded input, runs
`TemporalStreamBackbone.step`, then writes the output back to ID memory. The
memory update is an `nn.GRUCell` in `TemporalIdMemory`. A later event can
read a row just written by the previous event. The backbone state also spans
consecutive events in a lane. This feedback is the central serial dependency.

The isolated `ssm_lite` update is affine in its previous state:

`h_t = a_t * h_(t-1) + b_t`, where
`a_t = sigmoid(decay_proj(x_t))` and
`b_t = sigmoid(gate_proj(x_t)) * tanh(in_proj(x_t))`.

When every `x_t` is known before the sequence starts, `(a_t, b_t)` pairs
can be composed with an associative scan:
`(a_2,b_2) ∘ (a_1,b_1) = (a_2*a_1, a_2*b_1+b_2)`. In the **current full model**,
`x_t` includes memory read from preceding model outputs, so those pairs
cannot be computed up front. The nonlinear GRU memory transition cannot be
replaced by the same affine scan. Replacing only the backbone step with
Mamba would leave the event loop serialized. The existing Triton
`kernels/ssm.py` operation fuses one pointwise step, not a scan over time;
`backbone.type='mamba'` currently raises `NotImplementedError`.

Historical rich-context profiling measured a 102.7 ms scan phase and a
12.6 ms backbone subphase per measured batch, with rhythm/motif work in the
same scan. This is **not** a timing breakdown for the current cross-vehicle
configuration, where rhythm and motif are off. Early v2 phase records show
roughly 97–98% sampled GPU utilization during training; one raw
source-validation phase averaged about 19%. These are first-epoch
observations from running jobs, not final pilot results. Profile the actual
cross-vehicle batch before selecting a kernel or architecture.

### Priority A: exact acceleration of the current model

1. Finish the six v2 fits, evaluations, comparison, and coverage audit.
   Preserve their split manifests, phase summaries, and optimizer-step
   counts (expected two-epoch totals: 2,592 Silverado, 2,736 Forester,
   2,304 Traverse).
2. Profile a short warmed-up raw **and** behavioral training window on a
   V100. Separate data wait, encoding, ID-memory gather/context,
   `ssm_lite` step, memory GRU/write, heads/loss, backward, and optimizer.
   Use uninstrumented complete epochs for speed acceptance, since profiling
   synchronization can distort runtime. Measure kernel launches and memory
   traffic; high GPU utilization alone does not identify the bottleneck.
3. Optimize the measured hot path one change at a time: equivalent fused
   gathers/context, batched per-ID updates, or post-backbone writes. A
   fused trainable GRU update needs a verified backward path. Test
   `torch.compile` or custom kernels against the PyTorch reference. Do not
   silently fall back after failure. Any CUDA graph retry needs a full-fit
   memory preflight because replay previously exhausted V100 memory.
4. Compare event IDs, scores, losses, input/parameter gradients, state
   after each chunk, optimizer steps, and scored coverage. Include repeated
   IDs, source=destination, padding, reset, warm-up, and section boundaries
   in both feature modes. Require at least 20% faster scored events/s over
   a matched uninstrumented full epoch without changing coverage, steps,
   or exceeding device memory.

This path can reduce per-event overhead but cannot turn the full model's
nonlinear feedback into an ordinary parallel prefix scan while preserving
its exact outputs.

### Priority B: a new temporal variant that can scan over time

Remove learned per-ID memory from the temporal input path and retain causal,
precomputed behavioral features. The current feature table already includes
interarrival, same-ID timing, payload change, and first-seen signals. Static
normal-reference profiles may condition tokens but must remain fixed during
each query sequence. All temporal coefficients then depend on observed
token features and fixed reference context, not on earlier model outputs.
This is a **new model variant**, not an exact speedup of Priority A.
For a raw-input comparison, derive these causal features from each capture
with the same provenance and reset rules, and report raw and behavioral
results separately.

**B1: parallel `ssm_lite` first.** Compute all `a_t` and `b_t` over
`[lanes, time, hidden]`, then compose affine pairs with an associative
scan. Padding is the identity update `(a=1, b=0)`; handle nonzero initial
state, capture resets, and section boundaries. Preserve output projection,
residual, dropout, and normalization order. Keep the step path for online
inference. Compare full-sequence outputs, final state, chunked state carry,
input/parameter gradients, and streaming outputs with the sequential
reference. Start in FP32 and declare tolerances before mixed precision.
Measure scan-only and full-fit time, backward, memory, and padding.

**B2: official Mamba only after B1.** Pin a version/commit and verify its
V100 CUDA training path, causal convolution, backward, and streaming cache.
Upstream's [installation guide](https://github.com/state-spaces/mamba#installation)
currently separates the core package from an opt-in CUDA selective-scan
build; import success is insufficient evidence of a fast path. The
[Mamba paper](https://arxiv.org/abs/2312.00752) motivates an
input-selective parallel scan, while the
[Mamba-2 paper](https://arxiv.org/abs/2405.21060) describes a different
structured implementation. Choose by measured compatibility, full-fit
speed, memory, and quality on this workload. Verify both SSM and convolution
state in the streaming path and do not assume supported training through
arbitrary initial-state chunks.

**B3: measure the cost of removing learned memory.** Compare memory-on,
memory-off sequential, memory-off affine scan, and Mamba candidates at the
same split, scored coverage, optimizer-step budget, and checkpoint rule.
Choose architecture on source validation only. Evaluate held-out attack
tests after selection and repeat promising variants for more epochs/seeds;
the current two-epoch pilot cannot establish convergence. If learned
memory is essential to quality, retain Priority A or test a finite-lag
approximation under a distinct model name. Do not call the approximation
exact recurrence parallelization.

Adopt a parallel variant only if it passes sequence/step/gradient tests and
improves uninstrumented full-epoch scored throughput by at least 1.5× at
matched work. Report a fixed-wall-time comparison as well. Do not promote a
scan-only speedup that disappears in the full fit.

### Priority C: graph and reference context after the temporal gate

The original proposal's physical snapshot grid, graph, reference encoder,
forecasting heads, and suppression task form a broad representation and
objective change. They should not be prerequisites for the first speed
experiment. First add any graph as a same-time readout **after** the
parallel temporal layer, with no nonlinear graph feedback into that layer.
Events of different IDs are not synchronized by event index; a graph at a
common time needs explicit physical cutoffs and causal latest-state/age
joins. Compare any snapshot-grid model separately to the event-level model.

For target-reference conditioning, assign disjoint normal target files to
reference construction, threshold calibration, and normal testing before
training. Source-only transfer uses generic context with no target
statistics. Target-normal reference conditioning and target fine-tuning
are separate regimes. Never tune graph density, context size, checkpoints,
or thresholds on target attack labels. Include context-build time in
end-to-end measurements.

The remaining sections give design details for this optional extension.
They are **not** the critical path to parallelizing current-model training.

## 1. Follow-on measurements before changing the model

The code audit above locates preprocessing, temporal update, loss, streaming
inference, and split logic; there is no active GNN in the cross-vehicle model.
The remaining measurement is a cross-vehicle-specific profile of the full
forward/backward, memory feedback, optimizer, and data path. Record hardware,
PyTorch/backend versions, sequence length, lanes, padding, and state size.

Classify the custom recurrence: affine updates with coefficients computable from inputs/context can use a scan; nonlinear hidden-state-dependent transitions generally cannot use the same affine scan. Preserve the existing model and configuration. Reuse interfaces and naming already established in the repository.

## 2. Optional snapshot data contract and causal alignment

Raw records: `(vehicle_key, capture_key, timestamp, CAN_ID, DLC, payload,
label_if_available)`, plus channel when a dataset provides it. The current
CAN reader uses `arb_id` and `payload` after column normalization; it does
not require a channel field. Treat CAN ID (or `(channel, CAN_ID)` when
available) as a local stream identity, not a universal semantic label.
Vehicle keys manage context selection and splitting; exclude them from
learned features in the transfer configuration. Preserve observed
arbitration priority separately if investigating its value.

Split by vehicle and capture/time before window generation. Within each vehicle, designate disjoint reference, training/query, calibration, and test captures. All reference statistics and graphs must carry provenance. Never compute them from a full capture that includes future test traffic.

**If the optional graph/suppression track needs a common causal snapshot
grid**, choose a physical grid interval from training data, then freeze it
for the experiment. At each cutoff, aggregate only messages received by
that cutoff. Per stream/bin include count, arrival-gap summary, latest
payload bits/bytes with valid-byte mask, payload-change summary, time since
last arrival, and observed/missing flags. A bin with no messages is not
padding: its count and increasing age provide suppression evidence. Mask
unavailable payload targets. Define snapshot outputs as available after the
bin closes. Predict the next bin; score it once observed.

This yields `X: [B,T,N,F]`, `context_node: [B,N,C]`, `context_vehicle: [B,Cv]`, node/target masks, and a sparse graph per vehicle. Bucket node counts and pack real nodes where possible. Exclude padded nodes from pooling, graphs, losses, and scores. Keep sequences fixed length initially; reset state at capture boundaries and treat warm-up explicitly.

**Event-level alternative:** retain the current event-level data path and
causally gather each stream's latest hidden state at a common graph cutoff.
Never connect the kth event of one stream to the kth event of another: those
events need not be contemporaneous. Add age/missingness at graph cutoffs so
absent streams can still trigger alerts. This preserves within-bin detail.

## 3. Optional cross-vehicle behavior module

Expose a serializable `VehicleBehaviorContext` containing local stream index, feature schema, reference normalization, per-stream profiles, vehicle context, graph, calibration metadata, and reference provenance/version. Cache derived statistics; regenerate learned embeddings after encoder updates.

### Reference profiles and learned context

For each stream compute robust timing statistics, rate/jitter, payload-change frequencies, repetition, and optional counter-like descriptors. Do not assume bytes are decoded physical signals. Combine descriptors with embeddings of normal reference segments using a shared encoder and masked pooling to obtain `c_i`. Pool across nodes with an invariant set aggregator to obtain `c_v`. Keep profiles plus learned context as the first version; role slots and MoE are deferred ablations.

Learn context in reference/query episodes: build context from an earlier/disjoint normal reference, predict a separate query segment of the same vehicle, and share all encoder parameters across vehicles. Reference encoding can receive gradients during training. Freeze reference-derived context during each query sequence; it must not depend on query recurrent states or future observations. Do not use a learnable per-vehicle embedding as the default transfer mechanism.

### Vehicle graph

Infer a sparse graph from reference traffic alone. Start with top-K lagged predictive/dependence scores between activity and payload-change summaries; include self-information through residuals. Avoid treating correlation as physical causality. Handle constant streams and absent overlap explicitly. Store direction, strength, and lag where justified. Cap degree and benchmark graph density. A learned graph is a later experiment, constrained to reference data.

First deployment policy: freeze context, graph, and normalization. Unknown IDs receive a generic profile, shared node encoder, self path, and explicit unknown flag; define when a trusted reference refresh may add edges. Do not silently update reference profiles with attacked traffic. Report unknown-ID coverage and failures separately.

## 4. Optional Mamba-conditioned temporal core

Follow Priority B: first make the memory-free `ssm_lite` recurrence scan in
parallel, then consider a pinned official Mamba block if backend support and
profiling justify it. Check installed kernel availability rather than assuming
installation implies a fast GPU path. Provide a clearly labeled correctness
fallback, not an unnoticed slow production fallback.

Project causal features and context:

`u_i,t = Linear([X_i,t, c_i, c_v])`.

Use token-wise context projection initially so the standard fused block can remain intact. Optional FiLM conditioning must depend only on reference context and observed inputs, never the previous recurrent state. Use causal convolution, forward-only temporal processing, and token-wise normalization. No temporal normalization over future query positions.

Pack real streams as `[number_of_streams,T,D]` and run shared Mamba layers. A schematic selective update is `h_t = Abar_t h_(t-1) + Bbar_t u_t`; coefficients are input-derived and the state update remains affine. Preserve the block's stable transition parameterization. Do not equate learned Mamba step size with physical elapsed time; initially provide elapsed-time features explicitly. Explicit continuous-time discretization is a separate custom-SSM experiment.

Require full-sequence and streaming-step paths, carrying both SSM and convolution states. Reset both at capture/vehicle changes. Before supporting truncated chunks, verify the chosen backend supports initial/final states and the intended gradient semantics. Do not splice chunks using undocumented state handling.

## 5. Optional graph stage and prediction heads

Reshape temporal outputs to `[B,T,N,D]`. Apply a small shared sparse GNN separately at each physical cutoff using the vehicle graph; include context in node/edge projections. Batch independent graph snapshots with node offsets and no edges between vehicles or times. Chunk this graph batch when activation memory dominates. GNN parameters are shared, graph connectivity is local.

Pipeline: `reference -> context + graph`; `query -> conditioned Mamba -> same-time GNN -> heads`. Do not feed nonlinear graph outputs back into the same temporal recurrence. Stacking another temporal layer after a complete graph layer can remain parallel by layer, but defer it until the basic model is measured.

First heads predict next-bin count/arrival behavior, payload-change behavior, and optionally payload bits. Use probabilistic losses appropriate to each target: a count distribution, binary change probabilities, and masked bitwise probabilities. A relational head can predict a node using neighbor embeddings while excluding its own temporal embedding; compare with a self-only head. This helps establish whether graph information contributes rather than merely increasing capacity.

Do not use observed target-bin payloads to predict themselves. Forecast-only pretraining is the initial unsupervised objective; add a supervised attack head as a separately reported configuration. Suspiciousness is not proof of maliciousness.

## 6. Training and scoring for the optional graph/context model

Train across source vehicles with vehicle-balanced episode sampling so high-volume captures do not dominate. Normalize numeric features from authorized reference/training data; keep raw and relative timing features where normalization might hide attacks. Compare context-free, profile-only, learned-context, and full-context variants.

Sum masked temporal and relational losses with weights selected on source validation data. Report each component. For deployment, calibrate anomaly scores using disjoint normal calibration data; distinguish source-only thresholds from target-normal calibration. Aggregate per-node scores into alerts using a defined persistence/cooldown rule; raw maxima change distribution with node count, so measure or calibrate that effect.

Report two held-out-vehicle regimes separately: **no-target-reference transfer** using generic context/graph handling, and **normal-reference-conditioned transfer** with frozen weights but target context/graph construction. Optional target fine-tuning is a third regime. Never describe all three as equivalent zero-shot deployment.

## 7. Additional correctness and performance checks

- **Causality:** changing future query values cannot alter earlier predictions; test context, normalization, feature aggregation, and graph alignment too.
- **Sequence/step equivalence:** compare full-sequence outputs with streaming outputs and chunked outputs when supported; declare dtype-dependent tolerances. Carry convolution and SSM caches.
- **Scan correctness:** for a custom kernel compare outputs and parameter/input gradients against a short sequential reference. Verify mixed-precision stability and finite gradients.
- **Local-ID equivariance:** permute local stream tokens consistently with context and edges; outputs should permute correspondingly. Do not change physical arbitration features in this test.
- **Isolation/masking:** padding and other batch vehicles cannot affect a real node. Missing observations, suppression, and capture resets behave as specified.
- **Backend verification:** inspect profiler traces to confirm the intended optimized path. Benchmark warmed-up forward+backward+optimizer with accelerator synchronization, matched precision and comparable workloads. Report messages or bins/sec, peak memory, padding overhead, p50/p95 inference latency, and context-build cost.

Compare current memory-on `ssm_lite`, memory-off sequential `ssm_lite`,
memory-off parallel `ssm_lite`, and verified Mamba at matched work before
adding graph/context ablations. A matched-capacity TCN is a useful
nonrecurrent control. Measure end-to-end throughput as well as kernel speed.

## 8. Transfer experiments and reordered milestones

Evaluate leave-one-vehicle-out; manufacturer holdout where feasible. Investigate can-train-and-test provenance and comparable attacks before selecting it. Report per-vehicle/per-attack AUPRC, recall at fixed false-positive rate, aggregated false alarms/hour, time-to-alert, and normal-mode false alarms. Include timing-preserving manipulation/replay and suppression where available, not only flooding. Measure reference-duration sensitivity, unknown IDs, graph removal/rewiring, and context swapping between vehicles.

1. **Baseline audit:** finish v2 jobs and profile the current cross-vehicle
   feedback loop with raw and behavioral inputs.
2. **Exact optimization:** test the measured memory/context hot path with
   full-state, gradient, coverage, and end-to-end speed checks.
3. **Parallel core:** memory-off `ssm_lite` affine scan, then an official
   Mamba candidate only if its backend passes the V100 gate.
4. **Transfer quality:** compare with memory-on baselines at matched work,
   select on source validation, then evaluate held-out vehicles and repeat
   promising variants over more epochs/seeds.
5. **Optional behavior module:** reference profiles/context and fixed sparse
   graph, with independent graph/context ablations and declared target access.

Deliver repository-native modules/configs, state and context schemas,
meaningful tests, benchmark scripts, and a compact experiment report. Use
the project's conventions. The Priority A and B speed gates above are
predeclared; choose quality noninferiority bounds from source validation
before opening the held-out attack-test reports.

## Primary references to verify during implementation

- Mamba paper: https://arxiv.org/abs/2312.00752
- Official implementation and current backend requirements: https://github.com/state-spaces/mamba
- Mamba-2 / structured state-space duality: https://arxiv.org/abs/2405.21060
- Neural Relational Inference: https://arxiv.org/abs/1802.04687
- Conditional Neural Processes: https://arxiv.org/abs/1807.01613
- can-train-and-test: https://doi.org/10.1016/j.cose.2024.103777

These papers motivate components; they do not establish that this combined architecture generalizes CAN IDS across vehicles. That claim requires the held-out experiments above.
