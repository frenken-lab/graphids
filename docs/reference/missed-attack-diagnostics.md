> Historical snapshot diagnostics. Reproduce with `archives/snapshot-2026-10-16.tar.gz`; the associated launch scripts are retired.

# Source-only missed-attack diagnostics

`scripts/graph_mamba_missed_attacks.py` adds four diagnostic stages alongside the
production pipeline. It pins the selected mixed-attack epoch-6 checkpoint to
SHA256 `e513006088ffd06a6135a21dcc30fdf208ae1aed17ddbcdaeecb30c6ca86dcc6`.
The source manifest sidecar retains the reviewed source records and capture groups;
target metadata is checked for split integrity, but target files are never opened.
Production manifests, weights, and calibration artifacts remain unchanged.

```bash
run=/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-multiattack-source-v1
out=/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-missed-attacks-v1
python scripts/graph_mamba_missed_attacks.py audit \
  --source-run "$run" --config "$run/experiment.yml" \
  --checkpoint "$run/fit/checkpoints/best.ckpt" --output "$out"
torchrun --standalone --nproc_per_node=2 scripts/graph_mamba_missed_attacks.py extract \
  --source-run "$run" --config "$run/experiment.yml" \
  --checkpoint "$run/fit/checkpoints/best.ckpt" --output "$out"
torchrun --standalone --nproc_per_node=2 scripts/graph_mamba_missed_attacks.py probes \
  --source-run "$run" --config "$run/experiment.yml" \
  --checkpoint "$run/fit/checkpoints/best.ckpt" --output "$out"
python scripts/graph_mamba_missed_attacks.py report \
  --source-run "$run" --config "$run/experiment.yml" \
  --checkpoint "$run/fit/checkpoints/best.ckpt" --output "$out"
```

Run substantial processing in Slurm. The supplied CPU and two-A100 submission
scripts use the existing reviewed experiment and a separate diagnostic output
root. The GPU script also runs the regression checks and native distributed/serial
fixture. It waits for the CPU audit before fitting probes. Stage status files
and per-recording/per-epoch progress distinguish complete results from partial
sidecars. Audit reruns retain completed recordings. Probe reruns deterministically
refit their small heads from seed 42; no validation-based selection occurs.

The audit stores messages in 65,536-frame shards and embeddings in production
256-bin chunks. Frame shards retain derived CSV line, ID, bin, label, and causal
message features. The current bytes and validity masks preserve the within-bin
payload sequence, including multiple attacked IDs and legitimate overwrites.
Clock-row maps link changed/moved derived rows back to original rows. Other
rows retain their original line numbers. Observed ID-bin tables distinguish
attacker origin of the final frame from equality of its payload with an attacker
payload. Unknown frame labels remain unknown unless a positive label establishes
an ID-bin or bus-bin positive. Empty message bins score zero and are unsupported
for silence detection.

Graph hooks capture post-graph node states and the exact classifier input while
production `stream()` carries causal state across contiguous chunks. Complete
recording streams belong to one rank. Pooled states retain float32 precision;
node sidecars use float16 to bound storage. This small numerical difference is a
limitation of comparisons between pooled and localized probes. Frozen classifier
scores themselves retain production precision. Coverage verifies work on both
ranks, and the report checks conditioned validation scores against the retained
source replay.

The five readouts use Adam, learning rate 0.001, no weight decay, 4,096 supervised
units per optimizer step, and ten fixed epochs. Shards and units are shuffled
with seed 42. Continuous inputs are standardized using source `train_query`
units only. Balanced weights split mass equally by sign, then positive family,
vehicle and capture group (negative weights omit the family level). Related
variants share their capture group's allocation. The message network has one
64-unit SiLU layer. Epochs visit all known native supervised units, so optimizer
step counts differ between bus-bin, observed-ID and frame probes. The explicit-ID head uses observed ID-bin labels; the other
localized head learns max-aggregated bin predictions from bin labels.

The byte-transition baseline pools verified-normal source-training recordings
by CAN ID and byte position. It uses additive smoothing alpha 1 and marginal
probabilities for initial observations. Byte surprise is averaged across available
bytes, then max-aggregated across messages. Missing byte profiles/IDs are reported
separately. Because IDs are pooled across source vehicles, disagreement in ID
semantics can broaden this baseline's normal distribution. The fixed diagnostic
threshold is log(256); empirical normal-budget results provide a separate
comparison against the frozen forecast head.

`results.json` includes native-unit and common bus-bin metrics, family/vehicle
breakdowns, capture-group counts, and overwrite/novelty strata. Probe fixed
thresholds are 0.5. Existing detector thresholds are read unchanged. Budget
thresholds are selected on pooled empirical verified-normal query exposure for
each split and then applied unchanged to family and vehicle subgroups. Equal
scores are handled as ties under `>=`; achieved rates are reported, including
when ties force a conservative threshold. These exploratory sweeps are not a new
production calibration.

`scheduled-loss-contributions.json` recomputes checkpoint losses for the exact
training episode boundaries and context modes selected by each of the ten
schedules. Historical epoch metrics remain a separate artifact. Stream loss
sums/counts also include forecast targets across chunk boundaries. Forecast
training targets require independently verified normality; an attack recording
cannot contribute forecast gradients simply because some of its frames have
negative attacker labels.

Interpret current-feature collisions only for the input and context actually
examined. Shared payloads alone do not establish indistinguishability with
history or other IDs. Message/snapshot comparisons change both representation
and supervision. Attacker-origin attribution and operational anomaly detection
are different targets; no trusted incident intervals support incident recall,
attack duration or detection-delay claims here. Source probe gains are
exploratory evidence, not independent generalization results.

Primary validation comparisons reuse the retained selected-source scores after
checking replay coverage and scores. Applying the saved `CalibrationBundle` to
extracted forecast components in float64 reproduces the saved evaluator exactly;
online float32 normalization can differ in its last digits. Training forecast
scores use that same saved evaluator arithmetic. This changes no fitted
calibration parameter. `missed_attacks_report.sh` rebuilds the report on a CPU
allocation without repeating extraction or fitting. Completed coverage checks
are retained while their input and prediction files remain unchanged.

Read native frame metrics before interpreting an ID-bin or bus-bin improvement.
A high-scoring legitimate frame can share an ID-bin with an injected frame, so
max aggregation can improve bin detection while attacker-frame recall stays
zero. Such a result supports investigating contextual anomaly detection; it
does not establish origin attribution or operational incident detection.
Message comparisons also change the head's capacity alongside representation
and supervision. The completed study's recommendations are recorded in
[the evidence note](../empirical-notes/2026-10-16-missed-attack-diagnostics.md).

`embeddings/distributed-loss-contributions.json` attributes the frozen objective
using the original rank pairs and globally summed mask denominators. This is
separate from sums of local window means: normal-only forecast targets can exist
on only one rank of a step. Its numerators are reconstructed from retained
per-window means and source masks, with possible small floating-point rounding.
It does not reconstruct historical changing-weight training losses.
