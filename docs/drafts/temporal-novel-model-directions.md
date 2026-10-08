# Novel Temporal Model Directions for CAN Event Streams

This note sketches model directions that would be more distinctive than the
current temporal event classifier, causal attention baseline, GRU classifier,
and recurrent VGAE anomaly scorer. The goal is not to chase novelty for its own
sake: the strongest ideas below exploit the actual structure created by the
temporal preprocessing pipeline:

- events are continuous-time CAN message transitions;
- `src` and `dst` are consecutive arbitration IDs within a stream;
- `msg` contains payload bytes, byte deltas, inter-arrival time, entropy, and
  unknown-ID indicators;
- warmup events can initialize state without contributing to metrics;
- train/test source directories are held separate.

## Research Context

Continuous-time dynamic graph (CTDG) models are the closest family to this
representation. TGAT introduced temporal attention over time-encoded
neighborhoods for inductive temporal graph learning [Xu et al., 2020](https://arxiv.org/abs/2002.07962).
TGN then generalized many CTDG designs around event messages, per-node memory,
message aggregation, and embedding modules [Rossi et al., 2020](https://arxiv.org/abs/2006.10637).
Earlier recurrent interaction models such as JODIE learned dynamic embedding
trajectories for temporal interaction networks [Kumar et al., 2019](https://arxiv.org/abs/1908.01207),
while DyRep connected evolving node representations to temporal point processes
[Trivedi et al., 2018](https://arxiv.org/abs/1803.04051).

Recent benchmarks complicate the idea of a single state of the art. TGB argues
for realistic temporal splits and reproducible temporal graph evaluation, and
reports that simple methods can be very competitive on some dynamic node
property tasks [Poursafaei et al., 2023](https://arxiv.org/abs/2307.01026).
TGB 2.0 extends evaluation to temporal knowledge graphs and heterogeneous
graphs, emphasizing relation-type information, competitive simple heuristics,
and scalability failures on large datasets [Gastinger et al., 2024](https://arxiv.org/abs/2406.09639).
Transformer-style and mixer-style CTDG models such as DyGFormer and GraphMixer
target long temporal dependencies and efficient dynamic graph learning
[Yu et al., 2023](https://arxiv.org/abs/2303.13047),
[Cong et al., 2023](https://arxiv.org/abs/2301.12493). FreeDyG is another
useful reference because it explicitly adds frequency-aware modeling for
continuous-time dynamic graph link prediction [Tian et al., 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html).

State-space models are also relevant. Mamba uses selective state spaces for
linear-time sequence modeling and input-dependent state updates [Gu and Dao,
2023](https://arxiv.org/abs/2312.00752). Dynamic-graph Mamba variants are now
appearing, including DyG-Mamba, which frames dynamic graph modeling as long
sequence modeling with continuous state spaces [OpenReview, 2025](https://openreview.net/pdf/9140326d07557c09bd27ce8451624ceec0bfd37c.pdf).

## Direction 1: CAN-TGN With Arbitration-ID Memory

The cleanest next step is a TGN-style model specialized to CAN. Maintain a
memory vector for each known arbitration ID plus a shared unknown-ID memory. For
each event:

```text
event_i = (src_i, dst_i, t_i, msg_i)

message_i = f(
  memory[src_i],
  memory[dst_i],
  msg_i,
  time_encoding(t_i - last_seen[src_i]),
  time_encoding(t_i - last_seen[dst_i]),
)

memory[src_i], memory[dst_i] <- update(message_i)
score_i <- classifier(memory[src_i], memory[dst_i], msg_i)
```

Why it fits:

- CAN IDs recur with strong local and long-term regularities.
- The current preprocessing already supplies `src`, `dst`, `iat`, and
  `reset_after`.
- Warmup events can initialize memory without affecting metrics.
- Unknown IDs can share a memory plus hash-bucket embeddings.

Research angle: "event-memory temporal graph intrusion detection for CAN
streams." This is more graph-native than the current GRU because state is tied
to arbitration IDs, not only to the stream order.

Evaluation protocol:

- compare against event classifier, GRU, current causal attention, and VGAE;
- report whether per-ID memory improves rare attack recall and per-attack
  AUROC;
- ablate source memory, destination memory, and time encoding.

## Direction 2: Temporal Point-Process Anomaly Scorer

The current VGAE scores reconstruction surprise on `msg`. A stronger anomaly
model would estimate the likelihood of the next event:

```text
p(dst_id, payload_delta, iat | stream history, src_id)
```

The anomaly score becomes:

```text
score_i = -log p(dst_i) - log p(payload_delta_i) - log p(iat_i)
```

Why it fits:

- CAN attacks often alter timing, frequency, ID order, or payload transitions.
- Point-process dynamic graph work such as Know-Evolve and DyRep treats edge
  events as continuous-time processes whose intensities depend on evolving
  embeddings [Trivedi et al., 2017](https://arxiv.org/abs/1705.05742),
  [Trivedi et al., 2018](https://arxiv.org/abs/1803.04051).
- It gives a principled anomaly score rather than an arbitrary reconstruction
  error scale.

Implementation sketch:

- categorical head for `dst_id`;
- Gaussian or discretized mixture head for `iat`;
- regression or discretized byte-delta heads for payload changes;
- train only on `attack_free` data for anomaly claims.

This would be one of the most methodologically defensible anomaly models for
the project.

## Direction 3: CAN-Mamba State-Space Backbone

A Mamba-style model could replace the GRU in the current recurrent classifier:

```text
x_i = concat(msg_i, emb(src_i), emb(dst_i), time_features_i)
h_i = selective_state_space(x_i, h_{i-1})
score_i = head(h_i)
```

Why it fits:

- event streams are long;
- full attention over long CAN streams is expensive;
- GRUs are simple but can struggle with long-range dependencies;
- Mamba-style selective state spaces are designed for linear-time long sequence
  modeling [Gu and Dao, 2023](https://arxiv.org/abs/2312.00752).

The important caveat: Mamba is a sequence model, not automatically a relational
temporal graph model. The novel version should inject relational structure
explicitly through:

- source/destination ID embeddings;
- per-ID memory reads;
- stream reset handling;
- time-delta features;
- optional neighborhood summaries over recent transitions.

Research angle: "selective state-space modeling of relational CAN event
streams." The most convincing contribution would be a state-space backbone
that keeps the preprocessing contract intact and shows better long-context
scaling than causal attention or GRU.

## Direction 4: Memory + Mamba Hybrid

The most promising novel architecture is a hybrid:

```text
stream state:  h_stream <- Mamba(x_i, h_stream)
ID memory:     m[src_i], m[dst_i] <- TGN-style update
prediction:    head(h_stream, m[src_i], m[dst_i], msg_i)
```

This separates two kinds of temporal context:

- stream state captures recent rhythm, burstiness, and order;
- ID memory captures long-lived behavior of each arbitration ID.

Why this is stronger than plain GRU or plain TGN:

- a stream-level state can notice contextual attack phases;
- ID memory can notice when a normally stable ID starts behaving differently;
- the model remains causal and compatible with warmup/scored masks.

This is a good candidate for a paper contribution because it is not just
"apply Mamba to CAN"; it is a relational state-space model shaped by the
preprocessing.

## Direction 5: Frequency- and Rhythm-Aware Branch

CAN traffic has strong periodicity. Add a branch that summarizes recent timing
behavior for each ID or transition:

```text
features:
  rolling mean/std of IAT
  recent event count in time windows
  transition frequency A -> B
  spectral or Fourier time encoding
```

Then combine rhythm features with the event representation. FreeDyG is a useful
reference because it argues that frequency information can improve
continuous-time dynamic graph link prediction [Tian et al., 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html).

Why it fits:

- flooding, suppression, masquerade, and timing attacks often disturb message
  frequency before payload content becomes informative;
- rhythm features are interpretable;
- they can be added to both supervised and anomaly models.

This direction is less flashy than Mamba, but it may be more effective for CAN.

## Direction 6: Self-Supervised Pretraining on Attack-Free Streams

Use attack-free streams to pretrain an event encoder before supervised
fine-tuning or anomaly scoring:

```text
tasks:
  predict next arbitration ID
  predict next inter-arrival time
  reconstruct masked payload bytes
  classify whether a transition is temporally corrupted
  predict payload delta bucket
```

Why it fits:

- labeled attack events are sparse and imbalanced;
- attack-free streams are a natural source of normal behavior;
- learned normality can support both anomaly scoring and supervised
  fine-tuning.

This could be paired with TGN, Mamba, or the memory+Mamba hybrid.

## Direction 7: Transition-Motif Model

Instead of treating each event independently, model short transition motifs:

```text
A -> B -> C
B -> B -> D
unknown -> A -> A
```

A lightweight motif encoder can summarize the last `k` IDs and time gaps before
classification. This sits between pure sequence modeling and graph motif
analysis. It is especially attractive for CAN because many ECUs emit stable
message schedules.

Possible heads:

- supervised attack classifier;
- motif likelihood anomaly scorer;
- next-ID ranking head.

This direction is less tied to broad CTDG SOTA, but it is highly domain-aware
and easy to visualize.

## Recommended Priority

1. **Memory + Mamba hybrid** if the objective is a novel model contribution.
2. **Temporal point-process anomaly scorer** if the objective is a strong
   methodology for attack-free anomaly detection.
3. **CAN-TGN with ID memory** if the objective is a solid temporal graph
   baseline that is more graph-native than the current GRU.
4. **Frequency/rhythm branch** as a practical add-on to any of the above.
5. **Self-supervised pretraining** once the final supervised/anomaly baselines
   are complete.

The minimum publishable extension I would prototype first is the CAN-TGN. The
most interesting paper direction is the memory + Mamba hybrid, because it
combines a current sequence-modeling idea with a relational event-memory
structure that is specific to the CAN temporal preprocessing.
