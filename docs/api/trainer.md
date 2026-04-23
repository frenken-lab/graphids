# Core: Trainer

Pure-PyTorch training loop — Lightning was removed. Single-GPU only
(project targets 1× V100), handles AMP via ``GradScaler``, gradient
clipping, AMP-safe scheduler skipping on inf/nan scale-warmup batches,
and callback lifecycle using the same hook names as Lightning so the
OTel + curriculum callbacks ported over without change.

## `graphids.core.trainer`

::: graphids.core.trainer
