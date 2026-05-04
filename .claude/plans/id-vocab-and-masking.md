# CAN ID Vocab Scope, Embedding Masking, and Stack Direction

> Decision doc. Captures the literature on CAN-IDS arb_id handling, audits
> the current GraphIDS stack against it, and lists the concrete
> implementation changes that follow.

## Problem

CAN IDS must navigate three coupled tensions on the arb_id dimension:

1. **Benign novel IDs vs malicious novel IDs** — a different car model has
   legitimate ECUs the IDS was never trained on. These must NOT route to
   "attack" by default.
2. **Memorization vs generalization** — supervised classifiers trained on
   "arb_id 0x000 = DOS" inflate same-vehicle AUROC and collapse
   cross-vehicle. The hex value is a bad anchor; the *behavior* is the
   real signal.
3. **Heterogeneous attack semantics** — DOS is ID-anchored (the flooding
   ID is the signal); fuzzy / gear / RPM are largely ID-agnostic
   (statistical / semantic anomalies on payload + IAT).

A model that memorizes attack IDs aces same-vehicle benchmarks and fails
deployment. A model that ignores ID entirely loses the strongest signal
for ID-anchored attacks. The right answer is somewhere between, and
literature handles it with explicit techniques rather than data-layer
choices.

## Literature techniques

| Technique | Used in | Effect on the three tensions |
|---|---|---|
| **No ID embedding** (payload + IAT only) | CANet, GIDS, LSTM-AE family | Trivially generalizes (1, 2). Loses ID-anchored DOS signal (3). |
| **Hash encoding** (feature hashing of arb_id) | CANTransfer + variants | Cross-vehicle by construction (1). Collisions regularize (2). Mild discriminability hit (3). |
| **Per-ID model + ensemble** (one autoencoder per arb_id) | CANet | Sidesteps shared-vocab entirely. Expensive at runtime; doesn't help truly novel IDs. |
| **Embedding dropout / identity masking** | Standard NLP, not yet common in CAN IDS | Forces dual-path reasoning (identity + behavior). Trades small same-vehicle drop for large cross-vehicle gain. |
| **Per-attack-type eval** (DOS / fuzzy / gear / RPM separately) | ROAD-using papers | The diagnostic that exposes memorization. Same-vehicle DOS AUROC ~0.99, cross-vehicle ~0.6–0.8. |
| **Cross-vehicle holdout** (train Vehicle A, test Vehicle B) | ROAD, CANTransfer | The honest evaluation. Reveals the memorization tax. |
| **Domain adaptation / fine-tune** | CANTransfer | Adapts embedding table to target vehicle with few benign samples. Doesn't fix attack-ID memorization. |

The gap between "same-vehicle" and "cross-vehicle" numbers in the
ROAD-evaluation literature is the **memorization tax**. The literature
does not solve this; it papers over it with same-vehicle benchmarks.

## Current GraphIDS stack — direction audit

| Component | Verdict | Why |
|---|---|---|
| `BaseGraphSource._scan_vocab` (all-splits union) | **OPPOSITE DIRECTION** | Bakes the "same-vehicle, known-attack-set" assumption into the data layer. Test attack IDs get unique embedding rows, which is exactly what enables memorization. |
| VGAE (unsupervised reconstruction) | **SUFFICIENT** | Pure novelty detector; no attack labels in loss; can't memorize attack IDs by construction. Reconstruction error fires for both unknown-malicious and unknown-benign — correct one-class behavior. |
| GAT (supervised classifier) | **NEEDS EXTENSION** | This is where memorization lives. Embedding dropout / identity masking is the low-cost intervention. |
| Fusion | **SUFFICIENT (architecturally)** | Right layer to arbitrate VGAE behavior-score vs GAT identity-score. May need feature reweighting once GAT's identity dependence drops. |
| Test metrics (single AUROC) | **MISSING** | Per-attack-type AUROC needed to see the memorization tax at all. `attack_type` column is already in the dataset; just not surfaced. |
| Cross-vehicle eval | **MISSING** | HCRL ships sonata + chevy splits (`hcrl_sa`, `hcrl_ch`); the ablation isn't wired. |

The data layer is currently optimized for the inflated-numbers regime.
The model stack (VGAE + GAT + fusion) is structurally sound — VGAE
already does the right thing, fusion is the right shape — but GAT's
training procedure leaves memorization on the table and current
evaluation can't even detect it.

## Implementation plan

### Phase 1 — vocab scope (data layer)

- Add `vocab_scope: Literal["train", "all"] = "train"` to
  `BaseGraphSource`. Default flips to `"train"`: only train_dirs are
  scanned, novel test arb_ids route to UNK (index 0).
- This collapses the double CSV read (train Dataset's `process()`
  persists `vocab.json` from train rows; test Datasets read it). Source
  shrinks; `_scan_vocab` deletes.
- Existing caches built under all-splits vocab are NOT migrated —
  cache_key prefix gains `|voc:train` / `|voc:all` so the two regimes
  produce distinct caches and can coexist for ablation.

### Phase 2 — identity masking (model layer)

- Add `id_mask_rate: float = 0.15` to VGAE and GAT init args. During
  training (not eval), replace 15% of arb_ids with UNK *before*
  embedding lookup. Off by default at first to keep baseline numbers
  unchanged; flipped on for the dropout-arm runs.
- Implementation: drop into `forward()` before the `nn.Embedding(num_ids, ...)`
  call. ~5 LOC each. Symmetric to BERT-style token masking.
- VGAE's existing 15% node-token masking (mask_token / mask_id, see
  `vgae.py:191-200`) is a separate mechanism (mask the input feature for
  reconstruction); identity masking replaces the categorical id only.
  The two compose; can run together.

### Phase 3 — per-attack-type metrics

- `attack_type` column already populated in `data.attack_type`.
  Threshold-flavor models (VGAE, DGI) and classifier-flavor (GAT,
  fusion) both need per-attack-type AUROC at `on_test_epoch_end`.
- Group test_buffers by attack_type, compute AUROC per group + macro
  average, log alongside existing `test_auroc`.
- Logged as MLflow metrics: `test_auroc.dos`, `test_auroc.fuzzy`,
  `test_auroc.gear`, `test_auroc.rpm`, `test_auroc.macro`.

### Phase 4 — per-test-subdir reporting (cross-vehicle is structural)

`set_01..set_04` from Lampe & Meng's `can-train-and-test-v1.5` are
**already** a cross-vehicle / cross-attack ablation grid by design.
Each `set_0X/` ships five subdirs (PDF §5.5):

- `train_01/`
- `test_01_known_vehicle_known_attack/`
- `test_02_unknown_vehicle_known_attack/`
- `test_03_known_vehicle_unknown_attack/`
- `test_04_unknown_vehicle_unknown_attack/`

The four sub-datasets rotate which of the four vehicles (2011 Impala,
2011 Traverse, 2016 Silverado, 2017 Subaru Forester) and which attack
family fall into the "unknown" slots. The paper does not publish a
per-set vehicle assignment table; it must be read from each `set_0X/`
on disk.

Implications:

- **No `cross_vehicle.jsonnet` needed.** The 4-way test-subdir split
  IS the cross-vehicle ablation. The current per-test-subdir tensor
  layout in `BaseGraphSource.build()` already produces one
  `data_test_<subdir>.pt` per condition.
- **What's missing is reporting.** Today the test loop pools metrics
  across all test subdirs into one `test_auroc`. Phase 4 adds
  per-test-subdir AUROC tags so the (vehicle, attack) condition is
  surfaced — `test_auroc.test_01_known_vehicle_known_attack`,
  `test_auroc.test_02_unknown_vehicle_known_attack`, etc.
- **HCRL `sa↔ch`** stays as a separate cross-manufacturer datapoint
  outside the can-train-and-test grid. Its split layout is the older
  attack-free / with-attacks shape, not the 4-way unknown grid.

Ablation grid is now 2 × 2 × 4 (vocab_scope × id_mask_rate ×
test_subdir condition), per dataset:

- `vocab_scope ∈ {train, all}`
- `id_mask_rate ∈ {0.0, 0.15}`
- reported per `test_0{1,2,3,4}` subdir, with optional per-attack
  breakdown (Phase 3) within each subdir.

The Phase 1 (`vocab_scope="train"`) argument lands harder here: by
construction `test_02` and `test_04` contain a **held-out vehicle**.
Cross-manufacturer pairs (Chevrolet train → Subaru test) have near-zero
arb_id overlap (paper §3 lines 314, 374). So `vocab_scope="all"` is
literally enumerating the held-out vehicle's IDs into the embedding
table — exactly the leak the dataset's design is trying to prevent.
This makes `"train"` the only defensible default for any reported
`test_02` / `test_04` number.

## Expected ablation story

Hypothesis to falsify:

| Arm | Same-vehicle DOS | Cross-vehicle DOS | Cross-vehicle fuzzy |
|---|---|---|---|
| `vocab=all, mask=0` (current default) | very high | low | moderate |
| `vocab=train, mask=0` | high | low–moderate | moderate |
| `vocab=train, mask=0.15` | moderate–high | moderate–high | moderate–high |

The same-vehicle / cross-vehicle gap on DOS *is* the memorization tax.
The contribution is showing that a small samevehicle sacrifice (mask
on, train-only vocab) buys a real cross-vehicle gain. Per-attack
breakdown shows the win is concentrated on ID-anchored attacks where
memorization was doing the false work.

## Out of scope

- **Hash encoding** as an alternative to embeddings. Bigger
  architectural commitment; defer until Phase 1–4 numbers are in.
- **Domain adaptation / fine-tune on target vehicle**. Different paper
  scope; the Phase 4 ablation already establishes the baseline gap.
- **Per-ID-model ensemble** (CANet-style). Doesn't compose with the
  graph formulation.

## Decisions (resolved)

1. **`vocab_scope` default = `"train"`.** Resolved by mechanical argument
   (above). All-splits scope produces random, never-trained embeddings
   for test-only IDs and denies the model a trainable novelty path.
   Train-only + UNK collapses the same uncertainty into one explicit
   token, which becomes useful the moment Phase 2 is on. The
   can-train-and-test design (held-out vehicle in `test_02` / `test_04`)
   makes `"all"` actively wrong for those subdirs.
2. **`id_mask_rate` default = `0.0`** (off), wired as a kwarg from day
   one so a sweep over `{0.0, 0.10, 0.15, 0.30}` is one TLA flag away.
   First round reproduces current numbers; sweep follows once the
   per-test-subdir reporting is in place and there's a baseline to
   measure against.
3. **Cross-vehicle datapoints**: the can-train-and-test 4-way test-subdir
   split is the primary axis (already in the data). HCRL `sa↔ch`
   provides a separate cross-manufacturer point with the older split
   layout. No new plan jsonnet; the work is per-test-subdir metric
   reporting in Phase 4.

## Resolved metadata (can-train-and-test v1.5)

Source: paper (Lampe & Meng 2024, *Computers & Security* 142,
DOI 10.1016/j.cose.2024.103777, §4.1 + §5.5) plus bitbucket directory
inspection of `set_0X/{train_01_attack_free,train_02_with_attacks,test_0Y_*}`
contents (filename suffix encodes vehicle).

### Vehicle encoding

Filename suffix `-N` → vehicle N. Paper §4.1 lines 1859–1866:

| N | Vehicle | Body | Manufacturer |
|---|---|---|---|
| 1 | 2011 Chevrolet Impala | sedan | Chevrolet |
| 2 | 2011 Chevrolet Traverse | full-size SUV | Chevrolet |
| 3 | 2016 Chevrolet Silverado | pickup | Chevrolet |
| 4 | 2017 Subaru Forester | compact SUV | Subaru |

### Per-set vehicle + attack assignment

Read from filenames in each `set_0X/`:

| Set | Train vehicles | Held-out vehicles (test_02 / test_04) | Train attacks (in `train_02_with_attacks`) | Held-out attacks (in `test_03_known_vehicle_unknown_attack`) | Cross-mfr held-out? |
|---|---|---|---|---|---|
| set_01 | Impala + Traverse (Chevy only) | Silverado + Forester | DoS, force-neutral (gear), rpm | double-spoof, fuzzing, interval, speed, systematic | **YES** — Subaru held out |
| set_02 | Impala + Traverse (Chevy only) | Silverado + Forester | double-spoof, fuzzing, interval | DoS, force-neutral, rpm, standstill | **YES** — Subaru held out |
| set_03 | Silverado + Forester (mixed) | Impala + Traverse (Chevy only) | DoS, double-spoof, force-neutral | interval, rpm, rpm-accessory, speed, speed-accessory | NO — Subaru in train |
| set_04 | Silverado + Forester (mixed) | Impala + Traverse (Chevy only) | interval, rpm, rpm-accessory | DoS, double-spoof, force-neutral, fuzzing, triple-spoof | NO — Subaru in train |

So the four sub-datasets pair up:

- **set_01 + set_02** are the cross-manufacturer challenge: train on
  Chevy-only, hold out Subaru. The `test_02` / `test_04` numbers on these
  two sets are the IDs-don't-overlap headline.
- **set_03 + set_04** are the within-Chevrolet plus same-mfr-Subaru
  challenge: train includes the Subaru, hold out two Chevrolets. Easier
  generalization story (some manufacturer-shared IDs in training), but a
  cleaner "same vehicle family, different model year" comparison.

Per-set sample counts (paper §5.5 lines 2177ff): set_01 ≈10.7M train
samples, set_02 ≈17.3M, set_03 ≈12.0M, set_04 ≈9.5M.

### Subdir layout (v1.5 ≠ paper-described v1.0)

The published paper describes a 4-test-subdir layout. v1.5 on bitbucket
has six test subdirs + two training subdirs:

```
set_0X/
  train_01_attack_free/          ← attack-free CSVs only
  train_02_with_attacks/         ← attack-free + a subset of attacks
  test_01_known_vehicle_known_attack/
  test_02_unknown_vehicle_known_attack/
  test_03_known_vehicle_unknown_attack/
  test_04_unknown_vehicle_unknown_attack/
  test_05_suppress/              ← v1.5 addition
  test_06_masquerade/            ← v1.5 addition
```

GraphIDS's `configs/data/datasets.json` `train_subdir` /
`train_attack_subdir` / `test_subdirs` shape already matches this.
Action item: confirm `test_subdirs` in the catalog includes test_05 +
test_06, not just the original four.

### Attack taxonomy

Nine unique attacks (paper §4.3 lines 1925–1937), several with variants:

1. DoS
2. Combined spoofing — double, triple
3. Fuzzing
4. Gear spoofing (filename: `force-neutral`)
5. Interval
6. RPM spoofing — driving-mode (`rpm`), accessory-mode (`rpm-accessory`)
7. Speed spoofing — driving-mode (`speed`), accessory-mode (`speed-accessory`)
8. Standstill
9. Systematic

Plus v1.5 add-ons surfaced as test_05 / test_06 subdirs: suppress,
masquerade.

Current `ATTACK_TYPE_CODES` in `datasets/can_bus.py`:
```
normal/attack_free/benign → 0
dos → 1
fuzzy/fuzzing → 2
gear → 3
rpm → 4
flooding → 5
malfunction → 6
```

Coverage gaps for can-train-and-test files: `force-neutral` (gear
variant — would need to alias or extend), `double`, `triple`, `interval`,
`speed`, `speed-accessory`, `rpm-accessory`, `standstill`, `systematic`,
`suppress`, `masquerade`. Eleven new codes or aliases. Without this
extension, any per-attack metric (Phase 3) on can-train-and-test data
falls into `unknown_<int>` buckets and is uninterpretable.

### Implications for the plan

- **set_01 + set_02 are the high-value cross-manufacturer rows.** When
  reporting Phase 4 numbers, lead with these. set_03 + set_04 are
  weaker generalization tests because Subaru is in train.
- **`vocab_scope="train"` is mechanically correct on set_01/02.** The
  Subaru's arb_ids (test_02 / test_04) have near-zero overlap with the
  Chevrolet-only training vocab — `vocab_scope="all"` would build
  embedding rows for IDs the model literally cannot have learned.
- **Catalog update is prerequisite.** Before any Phase 1 cache
  rebuild lands, `configs/data/datasets.json` for `set_01..set_04`
  needs `train_subdir=train_02_with_attacks`,
  `train_attack_subdir=…` (verify what's there),
  `test_subdirs=[test_01_…, test_02_…, …, test_06_…]`.
- **Attack taxonomy widening is prerequisite for Phase 3.** Add the
  eleven new codes/aliases to `ATTACK_TYPE_CODES` and the inverse in
  `ATTACK_TYPE_NAMES`. Pure data; no model changes.
