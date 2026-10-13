> Historical temporal-stack notes. For current graph–Mamba implementation status and next-session work, read [NEXT_STEPS.md](NEXT_STEPS.md).

Yes. Now that the execution path is healthy and the 20-run fit matrix exists, the next step should not be “more jobs” yet. I’d move into a publication-readiness audit in this order:

**1. Validate Methodology**
The highest priority is confirming the experimental setup is scientifically defensible.

Focus on:
- Data split integrity: no temporal leakage across train/val/test.
- Preprocessing invariants: same vocab/ID mapping rules across splits, no test-derived statistics unless explicitly intended.
- Class imbalance handling: whether accuracy is misleading and whether AUROC/MCC/F1 per attack are the right primary metrics.
- Warmup handling: confirm `val_warmup_events` / `test_warmup_events` do not leak scored labels or distort evaluation.
- One-epoch baselines: decide whether these are just pipeline probes or legitimate baselines. For publication, they are probably probes only.

Deliverable: a short `methodology-audit.md` with pass/fail items and fixes.

**2. Define Final Evaluation Protocol**
Before training longer runs, lock the target protocol.

I’d define:
- Which datasets are in the main table.
- Which models are main baselines: event classifier, GAT, RNN, VGAE, maybe fusion later.
- Which metrics are primary: likely AUROC, MCC, macro F1, per-attack AUROC/F1, not accuracy.
- How many seeds are required. One seed is not enough for publication unless framed as a systems/prototype result.
- Whether results are fit-validation, test, or held-out test only. Publication tables should be test-stage.

Deliverable: `docs/empirical-notes/...final-eval-protocol.md`.

**3. Fix the Train/Test Artifact Story**
Right now the fit matrix is useful, but not final because these runs are `stage: fit`, one epoch, and checkpointing is not the main artifact story.

Next tighten:
- Enable/check checkpointing for final fit runs.
- Define test configs that consume exact checkpoints.
- Ensure MLflow records checkpoint path, git SHA, config, dataset version, preprocessing version.
- Make `gx exp results` views for final test results, not just fit probes.

Deliverable: checkpointed train + test pipeline for one model/dataset first.

**4. Resource/Efficiency Audit**
Do this before launching expensive final grids.

Measure:
- Runtime per model/dataset.
- GPU utilization if available from logs/system metrics.
- CPU/memory headroom.
- Cache hit/miss behavior.
- Whether Ray is adding useful value for single-worker jobs or just orchestration overhead.
- Whether GAT/VGAE jobs are bottlenecked by data loading, graph construction, or GPU compute.

Deliverable: choose final job sizes and walltimes based on measured numbers, not guesswork.

**5. Model/Architecture Review**
After methodology is stable, review whether each baseline is meaningfully comparable.

Questions:
- Is event classifier too strong because it directly sees temporal features the graph models do not?
- Is GAT using enough topology to justify itself?
- Is VGAE’s anomaly score well calibrated and consistently logged?
- Are RNN/GAT/event classifier using comparable parameter budgets?
- Are “small” models enough, or should final runs use named capacity tiers?

Deliverable: model card style notes for each baseline.

**6. Run Final Experiments**
Only after the above:
- Longer checkpointed fit runs.
- Test-stage runs.
- Multiple seeds.
- Final MLflow result views.
- Exportable result tables.

My recommended immediate next action: do **Step 1, methodology audit**, starting with split/preprocessing leakage. That is the highest publication risk. Runtime and architecture matter, but a leakage issue would invalidate everything downstream.


The best bet is not one model family alone. I’d bet on a **hybrid temporal normality model**:

**TGN-style per-ID memory + Mamba/SSM stream backbone + point-process prediction heads + rhythm features.**

Call it something like **CAN-MemSSM** or **Relational State-Space Event Memory**.

**Why this is the strongest SOTA candidate**
For CAN data, the signal is not only “is this event feature vector strange?” The signal is:

```text
Did this ID appear at the right time?
Did it follow the expected previous ID?
Did its payload delta match its normal rhythm?
Did this source stream enter an abnormal phase?
Did a normally stable arbitration ID start behaving differently?
```

A plain GRU captures stream order, but not persistent per-ID behavior. A plain TGN captures node/ID memory, but may miss long stream-level phases. A plain VGAE reconstructs features, but does not directly model event likelihood. A plain GAT/attention model can learn context, but is expensive on long streams and does not naturally maintain deployment-like state.

The hybrid should look like:

```text
event_i = (src_i, dst_i, t_i, msg_i)

x_i = encode(
  msg_i,
  emb(src_i),
  emb(dst_i),
  rhythm_features_i,
  memory[src_i],
  memory[dst_i]
)

stream_state_i = Mamba(x_i, stream_state_{i-1})

memory[src_i], memory[dst_i] = TGN_update(
  memory[src_i],
  memory[dst_i],
  stream_state_i,
  msg_i,
  iat_i
)

heads:
  supervised:      P(attack_i)
  next-ID:         P(dst_i | history)
  timing:          p(iat_i | history)
  payload-delta:   p(delta_i | history)
  anomaly score:   -log p(dst_i, iat_i, delta_i | history)
```

**Why this is better aligned with current evidence**
Recent temporal graph benchmarks do not show one universal winner. TGB leaderboards still show strong heuristic and DyGFormer-style results, while TGB 2.0 stresses that relation information and scalability matter and that simple baselines can remain competitive. Dynamic graph anomaly benchmarks report that continuous-time dynamic graph models are stronger than discrete snapshot models for edge anomaly detection. Mamba-style dynamic graph models are now emerging specifically because long temporal dependencies are a weakness of many CTDG models.

For your CAN setting, that points to a model that is:

- **continuous-time**, not snapshot-based;
- **relation-aware**, using `src -> dst` transition structure;
- **stateful**, so warmup and reset semantics matter;
- **long-context capable**, where Mamba/SSM helps;
- **normality-likelihood based**, not just reconstruction based;
- **rhythm-aware**, because CAN traffic is periodic.

**Most likely ranking**
1. **Memory + Mamba + likelihood heads + rhythm features**
   Highest upside. Best match to CAN periodicity, long streams, and relational transitions.

2. **TGN-style arbitration-ID memory + point-process anomaly head**
   Stronger and safer than Mamba if you want a more established temporal-graph baseline.

3. **Mamba stream model with supervised and self-supervised heads**
   Good long-context sequence model, but less graph-native unless you add ID memory.

4. **Frequency/rhythm-enhanced GRU or TGN**
   Less novel, but very likely to improve CAN results because attacks disturb timing.

5. **Current VGAE/GAT/RNN baselines**
   Useful baselines, but unlikely to be SOTA unless the dataset is easy.

**My recommendation**
Prototype **CAN-TGN first** as the credible temporal-graph upgrade, then extend it into the **memory + Mamba hybrid**. For final claims, use the hybrid as the novel model and keep CAN-TGN, GRU, VGAE, and temporal attention as baselines.

The key methodological claim would be:

> CAN intrusion detection benefits from modeling both stream-level temporal state and arbitration-ID-local memory, with anomaly scores derived from the likelihood of the next relational event rather than from isolated event reconstruction.
