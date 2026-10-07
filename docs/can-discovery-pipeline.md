# CAN discovery pipeline and module contracts

For the distinction between DBC annotations, candidate layouts, role hypotheses, relationship findings, and independently corroborated roles, see [CAN signal evidence vocabulary](can-signal-evidence-vocabulary.md). The DBC-assisted label-completion task and the label-free transfer task have separate evaluation protocols.

## Unit of analysis

A frame is one timestamped observation of a payload. Structural inference uses the **collection of frames for one `(bus, arbitration ID, payload length)` family**, preserving recording boundaries. A candidate field is a bit interpretation of that family; its value series contains one value per applicable observed frame. Message behavior uses per-ID arrival and whole-payload-change histories. Relationship discovery joins two or more such histories in time. Evidence is validated across independent recordings, not by treating adjacent frames or overlapping windows as independent replications.

The current CAN-D preparation uses `prepare_traces`: it retains only families present in every supplied reference capture, excludes recognized transport families, takes at most `max_ids` families, and samples at most `samples_per_capture` frames from each capture. These are explicit coverage limits, not a single-message evaluation. Protocol checks use unsampled frames. PicaCAN's `bit_flip_fields` likewise accumulates adjacent-frame flips within each recording and ID; its default considers all observed IDs unless capped. Neither method learns a field boundary from one frame.

Multiplexing inference experiments and their command-line runners have been retired. The active decoder does not infer mode selectors. DBC-backed fields may still carry a known multiplexing condition for reference matching or evaluation; that metadata is not a discovered selector. Historical multiplexing pilot notes remain in the empirical record.

## Stages and contracts

| Stage | Input | Output | Current implementation |
| --- | --- | --- | --- |
| Provenance and grouping | Recordings of timestamped frames | Audited `Capture`s and `(bus, ID, length)` families | `preprocessing.capture`, `preprocessing.traces.prepare_traces` |
| Reusable layout preparation | Fitting captures | Eligible families, observed protocol bits, and lazy unscored exhaustive layouts | `can_discovery.prepare.prepare_families` |
| Label completion case | Prepared families and one declared evidence track | Scored vehicle-role case artifacts and DBC evaluation | `label_completion.case` |
| Message behavior | Audited captures | Per-ID timing, payload-change, and motif evidence | `relationships.behavior` |
| Structural field inference | Per-family frame sequences | Nonoverlapping spans, alternative layouts, uncovered bits, support, method and parameters | `decoding.tokenization.tokenize`; `can_d`, `supervised_can_d`, `bit_flip`, `bycan` are selectable partition methods |
| Exhaustive field generation | One eligible message family | Lazy, unscored individual field layouts that may overlap | `decoding.tokenization.tokenize(method='exhaustive')` |
| Candidate generation | Structural results and unsampled protocol checks | `FieldRegistry.candidates`, `selected`, tokenizations, conflicts | `decoding.signals.discover_fields` |
| Candidate values | A candidate and frames of its family | Timestamped raw numeric observations, where byte order and signedness permit extraction | `decoding.field_values.extract`, `relationships.shared.signal_time` |
| Relationship evidence | Message histories or candidate value series | Typed timing, value, redundancy, predictive, or physical-factor evidence with support and controls | `relationships.behavior`, `relationships.physical` |
| Optional joint selection | Compatible candidates and independently validated relationship evidence | Ranked layouts and role hypotheses, retaining unknown and alternatives | `relationships.physical.candidate_graph`, `combination_ranker` (experimental) |
| Assessment | Frozen outputs and independent recordings; optional DBC truth opened after fitting | Separate coverage, layout, value, role, and relationship measures | Standalone benchmark scripts |

`FieldRegistry` is the shared structural handoff. `candidates` is the hypothesis pool; `selected` is one conservative, nonoverlapping view. Its `FieldCandidate` identifies a local family and interpretation by bus, ID, payload length, bit start/width, byte order, signedness, and optional multiplexing condition. `tokenizations` preserves structural alternatives and uncovered bits. A physical `role` or label hypothesis is separate from those coordinates. The current tokenizer does not infer multiplexing; a null mode is an honest absence of that evidence. Consumers must not mistake `selected` for verified DBC truth or silently discard `candidates` when testing alternatives.

## Valid compositions

1. **Message behavior only:** fit timing and payload-change relations from captures without decoding fields.
2. **Structure only:** run a tokenizer and `discover_fields`; assess boundaries and raw values without any graph.
3. **General relationship discovery:** use candidate values, message behavior, or both; keep the evidence level on each relation.
4. **Joint discovery:** use relationships validated on source recordings to rerank compatible field and role hypotheses. Compare against structure-only and unary-role baselines.
5. **PicaCAN baseline:** use its own bit-flip extraction, TECG generation, and fixed SECG matching in `experiments.picacan`. Replacing its extraction with CAN-D is a separately named hybrid experiment, not a paper reproduction.

PicaCAN's temporal graph and the package's general relationship modules answer related but different questions. PicaCAN groups extracted physical-signal candidates with delayed Pearson correlation, estimates directed TECG edges with RatioDSTE, then matches a declared ICE graph. General message behavior can run before field extraction; physical factors can test specific candidate roles without requiring PicaCAN's fixed five-node graph. The combined chassis graph is another separate experiment.

The stages are **composed in a full discovery run**. Their separate interfaces permit independent accuracy measurements, ablations, and use when a later stage lacks evidence. PicaCAN first extracts candidates within each ID, then groups candidate signal series across all processed IDs into semantic nodes, estimates edges among those nodes, and searches the resulting graph for a five-node ICE-role mapping. It does not test one ID at a time against the entire powertrain model. A message ID may supply multiple candidate fields; a semantic node may group highly correlated fields from different IDs. The current baseline processes all observed IDs on the selected capture bus by default, with `max_ids` only as an explicit cap.

## Evidence and evaluation rules

- Keep four claims separate: candidate **coverage**, selected **bit layout/value accuracy**, **physical name** accuracy, and **relationship** replication. A predictive relation does not verify either endpoint's physical name or establish causality.
- Preserve recording identity, source hashes, bus, family key, sample counts, method, parameters, missing/constant bits, and reasons for abstention. Report both per-family support and independent-recording support.
- Fit on declared normal references. Freeze candidates, pivots, lag choices, and ranker weights before checking development or sealed recordings. A target DBC is post-fit truth unless a protocol explicitly declares an oracle or source-calibration condition.
- For cross-ID relations, align in physical time and retain observation age/masks; compare against target own-history and shifted or schedule-preserving controls. A within-ID boundary test and a cross-ID relation test have different denominators.
- Missing activity, no overlap, uncertain byte order, absent roles, and observationally equivalent layouts produce `unknown` or `unassessable`, not forced signals.

The standalone normal-profile runner was retired. Structural decoding, relationship benchmarks, integrated discovery, and the current label-completion workflow use the shared stage modules directly. Historical version-11 profiles remain research records. The PicaCAN runner and graph benchmark scripts remain isolated so their method and evaluation claims stay identifiable.
