"""Knowledge distillation as drop-in loss modules.

Two shapes because student/teacher KD has two genuinely different forms
in this repo:

- :class:`SoftLabelDistillation` — Hinton response-based KD. Wraps any
  scalar classification loss. The teacher takes the same input as the
  student and returns logits; the wrapper adds a temperature-softened KL
  divergence on top of the base loss. Used by GAT supervised training.

- :class:`FeatureDistillation` — feature-based KD for autoencoders. Wraps
  :class:`~graphids.core.losses.autoencoder.VGAETaskLoss`. The teacher
  runs its own forward pass and exposes intermediate ``z`` (latent) and
  ``cont_out`` (reconstruction) tensors; the wrapper adds MSE alignment
  on those features. Used by VGAE curriculum training.

Both classes are plain ``nn.Module``. They're built like any other loss —
via a class_path block in the rendered_config's ``model.init_args.loss_fn``,
instantiated by :func:`graphids.orchestrate._instantiate`, and passed into
the student as ``loss_fn``. No trainer plugin, no callback, no IO. The teacher is
held on CPU by default and moved to the student's device lazily inside
``forward`` — ``nn.Module`` storage is bypassed (``__dict__`` assignment)
so ``.to()``/``.cuda()`` doesn't try to shuttle it around.

Both expose ``last_hard_loss`` / ``last_soft_loss`` (or
``last_task_loss`` / ``last_kd_loss``) after each ``forward`` call so the
student module can log the two components separately without having to
re-run either pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphids._fs import atomic_load
from graphids.core.models.base import strip_orig_mod_prefix


def _load_teacher(ckpt_path: str, model: nn.Module) -> nn.Module:
    """Load checkpoint weights into ``model`` in-place and return it.

    Models that use lazy ``_build()`` (VGAE, GAT) are not constructed until
    ``num_ids`` is known.  When the teacher is instantiated from the plan spec
    it gets ``num_ids=0``, so ``_build()`` never fires and layer attributes like
    ``_uses_edge_attr`` are missing.  We pull the resolved hparams from the
    checkpoint and trigger ``_build()`` before loading weights.
    """
    ckpt = atomic_load(ckpt_path, map_location="cpu", weights_only=True)
    if hasattr(model, "_built") and not model._built:
        hp = ckpt.get("hyper_parameters", {})
        for k in ("num_ids", "in_channels", "num_classes"):
            if k in hp:
                setattr(model, k, hp[k])
                model.hparams[k] = hp[k]
        model._build()
        model._built = True
    state = strip_orig_mod_prefix(ckpt.get("state_dict", ckpt))
    remap = {k.replace("_orig_mod.", ""): k for k in model.state_dict()}
    state = {remap.get(k, k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


def _attach_teacher(module: nn.Module, teacher: nn.Module) -> None:
    """Hold ``teacher`` on ``module`` without it showing up in ``_modules``.

    Lightning's ``.to(device)`` walks ``self._modules`` and auto-transfers
    every child. We don't want the teacher on-device permanently — only
    during the brief forward pass — so we park it in ``__dict__`` which
    ``nn.Module.__setattr__`` would otherwise route into ``_modules``.
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    module.__dict__["teacher"] = teacher


def _run_teacher_on(student_device: torch.device, teacher: nn.Module, *args, **kwargs):
    """Move teacher to ``student_device``, run forward under ``no_grad``, move back."""
    teacher.to(student_device)
    try:
        with torch.no_grad():
            return teacher(*args, **kwargs)
    finally:
        teacher.to("cpu")


class SoftLabelDistillation(nn.Module):
    """Hinton soft-label KD wrapper for classification losses.

    ``loss = α · KL(softmax(student/T), softmax(teacher/T)) · T²
             + (1 − α) · base_loss(logits, labels)``

    Parameters
    ----------
    base_loss:
        Any ``nn.Module`` with the classification contract
        ``forward(logits, labels, graph=None) → scalar``. Typical choices:
        :class:`~graphids.core.losses.classification.CrossEntropyLoss`,
        :class:`~graphids.core.losses.classification.FocalLoss`.
    teacher_model:
        A bare (unweighted) teacher model, already instantiated by
        ``_instantiate`` from the ``teacher_spec`` plan key. Weights are
        loaded from ``teacher_ckpt_path`` in ``__init__``. Held off
        Lightning's auto-transfer path.
    teacher_ckpt_path:
        Path to the teacher checkpoint. Loaded on CPU inside ``__init__``.
    temperature, alpha:
        Standard Hinton KD hyperparameters.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        teacher_model: nn.Module,
        teacher_ckpt_path: str,
        *,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.temperature = temperature
        self.alpha = alpha
        teacher = _load_teacher(teacher_ckpt_path, teacher_model)
        _attach_teacher(self, teacher)

        # Populated on every forward() call so the training module can log them.
        self.last_hard_loss: torch.Tensor | None = None
        self.last_soft_loss: torch.Tensor | None = None

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        graph=None,
    ) -> torch.Tensor:
        hard = self.base_loss(student_logits, labels)
        teacher_logits = _run_teacher_on(student_logits.device, self.teacher, graph)

        T = self.temperature
        soft = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        self.last_hard_loss = hard.detach()
        self.last_soft_loss = soft.detach()
        return self.alpha * soft + (1 - self.alpha) * hard

    def log_components(self, model, *, batch_size: int, prefix: str = "") -> None:
        """Log per-component losses from the most recent ``forward`` call."""
        for name, value in (
            ("hard_loss", self.last_hard_loss),
            ("soft_loss", self.last_soft_loss),
        ):
            if value is not None:
                model.log(f"{prefix}{name}", value, batch_size=batch_size)


class FeatureDistillation(nn.Module):
    """Feature-based KD wrapper for VGAE reconstruction loss.

    ``loss = α · (latent_w · MSE(project(z_s), z_t) + recon_w · MSE(cont_s, cont_t))
             + (1 − α) · base_loss(student_outputs, batch)``

    Parameters
    ----------
    base_loss:
        Typically :class:`~graphids.core.losses.autoencoder.VGAETaskLoss`.
        Contract: ``forward(student_outputs, batch) → scalar``. The
        wrapper forwards the exact same arguments to it.
    teacher_model:
        A bare (unweighted) teacher VGAE model, already instantiated by
        ``_instantiate`` from the ``teacher_spec`` plan key. Weights are
        loaded from ``teacher_ckpt_path`` in ``__init__``. Must accept
        positional args ``(x, edge_index, batch_idx)`` plus kwargs
        ``edge_attr=`` and ``node_id=``, matching
        ``GraphAutoencoderNeighborhood.forward``.
    teacher_ckpt_path:
        Path to the teacher checkpoint. Loaded on CPU inside ``__init__``.
    latent_weight, recon_weight, alpha:
        Dual-signal KD weights + convex combination weight.
    projection:
        Optional ``nn.Linear`` mapping student latent dim → teacher latent
        dim when they differ. Lives inside this module so it's on the
        auto-transfer path and gets its own gradients.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        teacher_model: nn.Module,
        teacher_ckpt_path: str,
        *,
        latent_weight: float = 1.0,
        recon_weight: float = 1.0,
        alpha: float = 0.7,
        projection: nn.Linear | None = None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.latent_weight = latent_weight
        self.recon_weight = recon_weight
        self.alpha = alpha
        self.projection = projection
        teacher = _load_teacher(teacher_ckpt_path, teacher_model)
        _attach_teacher(self, teacher)

        # Populated on every forward() call so the training module can log them.
        self.last_task_loss: torch.Tensor | None = None
        self.last_kd_loss: torch.Tensor | None = None

    def forward(self, student_outputs: tuple, batch, mask=None) -> torch.Tensor:
        task = self.base_loss(student_outputs, batch, mask=mask)

        cont_out, _canid, _nbr, z, _kl, _edge = student_outputs

        t_cont, _, _, t_z, _, _t_edge = _run_teacher_on(
            batch.x.device,
            self.teacher,
            batch,
        )

        z_s = self.projection(z) if self.projection is not None else z
        min_n = min(z_s.size(0), t_z.size(0))
        latent_kd = F.mse_loss(z_s[:min_n], t_z[:min_n])

        min_r = min(cont_out.size(0), t_cont.size(0))
        recon_kd = F.mse_loss(cont_out[:min_r], t_cont[:min_r])

        kd = self.latent_weight * latent_kd + self.recon_weight * recon_kd

        self.last_task_loss = task.detach()
        self.last_kd_loss = kd.detach()
        return self.alpha * kd + (1 - self.alpha) * task

    def log_components(self, model, *, batch_size: int, prefix: str = "") -> None:
        """Log per-component losses from the most recent ``forward`` call."""
        for name, value in (
            ("task_loss", self.last_task_loss),
            ("kd_loss", self.last_kd_loss),
        ):
            if value is not None:
                model.log(f"{prefix}{name}", value, batch_size=batch_size)
