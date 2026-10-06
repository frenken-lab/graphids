# CAN decoding and relationship discovery: research aims

## Aim and publication claim

The first publication should study whether an iterative, uncertainty-preserving process can recover useful CAN field structure and reproducible relationships from normal recordings. Its contribution is the **discovery and evaluation method**, including negative results and conditions under which evidence is insufficient. Attack detection and a GraphIDS Graph Mamba model are later studies; neither is needed to establish a discovery result.

The central hypothesis is that candidate decodings make value relationships testable, while independently replicated relationships can help distinguish otherwise plausible decodings. This must be tested against the simpler alternative: unary field evidence alone. Arrival timing and whole-payload changes provide a parallel relationship track when field decoding is unresolved.

The paper should answer three questions:

1. **Decoding:** Which active bit layouts, protocol fields, and raw values can be recovered on held-out recordings without using the target DBC during fitting? Where do multiple layouts remain observationally equivalent?
2. **Relationships:** Which timing or value dependencies improve prediction beyond each stream's own history, survive shifted or schedule-preserving controls, and replicate across independent recordings or vehicles? Under which operating contexts do they fail?
3. **Joint discovery:** Does validated relationship evidence improve exact-layout selection or named-role retrieval over the current decoder and unary-role baselines, without harming assessed vehicles? When it does not, does abstention preserve a more accurate account of uncertainty?

These are separate claims. Exact layout, held-out value equivalence, physical naming, relationship replication, and cross-vehicle transfer need separate denominators and evidence. A predictive edge does not establish causality; a relationship between proposed fields does not prove their physical names.

## Iterative research pipeline

1. **Freeze provenance and splits.** Parse frames with timestamp, ID, payload length and validity, recording identity, and capture hashes. Reserve distinct normal recordings for fitting and evaluation. Use source-vehicle DBCs only for declared calibration and the target DBC only for post-fit assessment. Treat shared sessions and duplicate files as one source of evidence.
2. **Propose, retain, and measure decodings.** Generate bounded alternative field spans, byte orders, signedness, and protocol hypotheses. Keep a conservative selected layout and the candidate pool distinct. Record unsupported bits, budget overflow, and null or unknown roles instead of forcing a physical interpretation.
3. **Mine relationships at the available evidence level.** Test arrival and payload-change relations even without decoded fields. For credible candidate values, use causal time alignment, observation and age masks, context, lag, and target own-history prediction. Compare against appropriate time-shift or schedule-preserving controls. Count support by independent recordings and vehicles, not overlapping windows alone.
4. **Feed evidence back into selection.** Use only relationships that pass source-side validation to rerank compatible candidate interpretations. Preserve competing layouts and cap redundant evidence from one underlying relationship. Never let target DBC labels or held-out outcomes enter the fitted ranking artifact.
5. **Audit the next iteration.** Evaluate selected layouts, candidate coverage, raw-value errors, role ranks, relationship replication, runtime, and failures on held-out data. Diagnose whether a miss came from candidate generation, insufficient motion or observability, alignment, ranking, or an unsupported physical assumption. Change one of those components on development data, then repeat the frozen evaluation protocol.

The loop is a research program, not a license to tune repeatedly on the same held-out trip. Changes motivated by a held-out failure need fresh independent evaluation or must be labeled exploratory.

## Current evidence and next gates

The [version 7 decoder and version 9 graph benchmark](../docs/empirical-notes/2026-10-04-candidate-graph-v9/REPORT.md) provide the immediate comparison. Across four held-out vehicles, the one-layout graph ties version 7 on exact-layout recall and named-role retrieval. The eight-layout graph loses aggregate exact-layout recall and breaches the per-vehicle precision limit. Neither passes the existing promotion gate. Source validation found too little replicated temporal or physical evidence to give those graph families positive weight in these folds. More graph machinery alone is therefore not a publication result.

The earlier [compositional benchmark](../docs/empirical-notes/2026-10-04-compositional-can-v5.md) shows that a broad pool can contain many correct active layouts while selected layouts and physical names remain weak. The [version 6 physical-factor audit](../docs/empirical-notes/2026-10-04-physical-v6.md) finds only speed–acceleration transferable under its two-vehicle source gate; other proposed factors remain discovery evidence. The [motif pilot](../docs/empirical-notes/2026-10-04-motif-pilot.md) also warns that repeated binary change patterns can merely encode schedules.

Prioritize the following questions before promoting relationship-based ranking:

- Can richer independent driving regimes and source recordings distinguish competing field boundaries and roles? Report when the available data cannot identify an exact width or physical name.
- Which relationships retain predictive gain after own-history, rate or schedule controls, and held-out context checks? Separate timing edges from value and physical-role factors.
- Does a source-validated graph improve the existing version 7 promotion measures across held-out vehicles? Report per-vehicle losses, abstentions, and computation alongside aggregate gains.

A publication can report a well-characterized negative answer. It should claim physical semantics only for fields supported by independent DBC, diagnostic, sensor, or controlled-maneuver truth. CAN-Train-and-Test message-ID anchors alone do not validate bit layouts or field names.

## Path to GraphIDS Graph Mamba

The discovery output should be a versioned, provenance-bound normal-reference artifact: candidate fields and alternatives, verified protocol rules, relationship type and lag, support and controls, observation masks, and uncertainty or unknown status. The artifact is an **input hypothesis** for model experiments, not a detector score. Freeze it before each query recording; build it only from the normal data allowed by that transfer regime.

The later [Graph Mamba plan](../vehicle-graph-mamba-joint-plan.md) can test three levels separately: raw timing and payload behavior, uncertain behavioral roles and validated relations, and verified physical signals where privileged truth exists. Compare each with the same model without discovery features and with shuffled or self-only graph controls. Report source-only zero-shot and target-normal-reference-conditioned transfer separately, including context cost and failure cases. A model gain establishes utility for detection under that evaluation contract; it does not retroactively validate a decoded field name or causal edge.

This handoff is warranted when the discovery artifact is stable across disjoint normal recordings, improves its own held-out discovery measures or honestly encodes uncertainty, and can be constructed without target-query or attack-label leakage. If named roles remain weak, pass timing, payload, and provisional relationship evidence with masks rather than invented physical labels.
