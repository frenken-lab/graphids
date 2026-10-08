# Investigation brief: One CAN IDS model across multiple vehicles

**Objective.** Investigate whether our graph/time-series CAN IDS can share detection knowledge across vehicles despite proprietary CAN IDs and payload layouts. Compare explicit vehicle-specific mapping/adapters with mixture-of-experts (MoE). Begin with a repository and literature audit; propose the smallest decisive experiment before implementing a large architecture.

## Questions to resolve

1. Distinguish **multi-vehicle learning** (training vehicles), **adaptation** (target normal traffic or labels available), and **zero-shot transfer** (unseen vehicle; no target fitting). State exactly which claim each experiment supports.
2. Determine whether current ID embeddings, graph construction, payload processing, normalization, and thresholds depend on vehicle identity or vocabulary. Identify unseen-ID behavior and input/output shape constraints.
3. Separate ID namespacing, `(vehicle, ID) → token`, from semantic alignment, `(vehicle, ID, bit layout) → common signal/role`. Namespacing prevents collisions but does not align signals. Can behavioral alignment work without DBCs?

## Candidate architectures

| Candidate | Shared component | Vehicle-specific component / hypothesis |
|---|---|---|
| Separate models; pooled model | Baseline comparison | Quantify benefits and negative transfer from sharing |
| Mapping + shared encoder/head | Temporal/GNN backbone and detector | Explicit vehicle-ID namespace; semantic mappings only where verified |
| Shared backbone + small adapters | Most parameters | Vehicle input adapters or conditioning; test against head-only personalization |
| MoE | Encoder and shared fallback expert | Compare known-vehicle routing with learned routing; test vehicle versus behavioral specialization |
| Behavior-conditioned GNN | Node encoder, message passing, scorer | Per-vehicle normal reference; local IDs identify nodes without requiring global ID semantics |

For behavioral features, examine per-ID period/jitter, payload change rates, repetition, counter-like patterns, and temporal relationships. A permutation-equivariant GNN handles node ordering, but numeric ID embeddings can still encode vehicle identity. MoE routing does not itself resolve payload semantics; inspect routing stability under attack and behavior on unknown vehicles.

## Evaluation requirements

- Use leave-one-vehicle-out evaluation; hold out manufacturers if data permit. Investigate **can-train-and-test** first because it offers comparable attacks across vehicles. Verify vehicle identities, capture provenance, and attack availability.
- Split by trip/capture/time **before** making overlapping windows. Fit preprocessing, graph relationships, hyperparameters, and thresholds only on authorized training/calibration data. Separate target adaptation captures from target test captures.
- Report three regimes separately: zero-shot; adaptation with normal target traffic only; limited labeled adaptation. Define whether parameters, reference statistics, graphs, or thresholds change in each regime.
- Report per-vehicle/per-attack AUPRC, recall at a fixed low false-positive rate, false alarms per hour with a defined alert aggregation rule, detection latency, and adaptation/compute costs. Separate flooding/fuzzing from replay, suppression, and timing-preserving payload manipulation where available.
- Ablate raw numeric IDs, namespaced embeddings, timing-only, payload behavior, graph edges, adapters, and routing. Test ID relabeling consistently; preserve arbitration priority as a separate feature when relevant. Examine whether experts merely memorize dataset/vehicle fingerprints.

## Evidence and deliverable

Audit ACHILLES (§III; Tables I–IV), CANPerFL (https://doi.org/10.3390/app13116369), E–UniCon (https://doi.org/10.1016/j.knosys.2025.114716), and can-train-and-test (https://doi.org/10.1016/j.cose.2024.103777). Read primary methods/code; record target-data access and distinguish adaptation from zero-shot claims. ACHILLES motivates statistical transfer, not demonstrated semantic alignment.

Return a concise recommendation with repository touchpoints, verified literature evidence, data gaps, and a minimal experiment matrix. Recommend the simplest architecture that shows measurable shared knowledge; specify success criteria and failure interpretations before training. Treat architecture benefits as hypotheses, not established results.
