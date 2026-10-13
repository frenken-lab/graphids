# Graph Mamba model

`vehicle_graph_mamba_joint` is the sole active model primitive. Its shared
parameters support varying local ID sets without learned raw-ID or vehicle
embeddings. Classification and next-bin forecasting share temporal Mamba and a
bounded graph readout; component scores remain separate.

See [training and evaluation](../reference/graph-mamba.md) for architecture,
causality, masks, and checkpoint migration.

::: graphids.core.models.graph_mamba

::: graphids.primitives_models
