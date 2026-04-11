"""SLURM commands: probe-budget."""

from __future__ import annotations

from typing import Annotated

import typer

from graphids._otel import get_logger, get_meter
from graphids.cli.app import (
    _complete_conv_type,
    _complete_dataset,
    _complete_model_type,
    _complete_scale,
    app,
)

log = get_logger(__name__)
meter = get_meter("graphids.budget")
_bpn_gauge = meter.create_gauge("budget.bytes_per_node", description="VRAM bytes per graph node")
_budget_gauge = meter.create_gauge("budget.max_nodes", description="Max nodes per batch")
_bwd_gauge = meter.create_gauge("budget.backward_multiplier", description="Backward/forward VRAM ratio")


@app.command("probe-budget", rich_help_panel="SLURM")
def probe_budget(
    dataset: Annotated[
        list[str] | None,
        typer.Option(help="Dataset(s) to probe", autocompletion=_complete_dataset),
    ] = None,
    model_type: Annotated[
        list[str] | None,
        typer.Option(help="Model type(s) to probe", autocompletion=_complete_model_type),
    ] = None,
    scale: Annotated[
        list[str] | None,
        typer.Option(help="Scale(s) to probe", autocompletion=_complete_scale),
    ] = None,
    conv_type: Annotated[
        list[str] | None,
        typer.Option(help="Conv type(s) to probe", autocompletion=_complete_conv_type),
    ] = None,
    lake_root: Annotated[str | None, typer.Option(help="Lake root path")] = None,
    dry_run: Annotated[bool, typer.Option(help="Print plan without probing")] = False,
) -> None:
    """Measure VRAM budget across (model × scale × conv_type × dataset). Requires GPU."""
    import torch

    from graphids.config.constants import LAKE_ROOT, VALID_MODEL_TYPES, VALID_SCALES
    from graphids.config.topology import cache_dir, data_dir, dataset_names
    from graphids.core.data.budget import node_budget
    from graphids.core.models.factory import build_model_from_spec

    if not torch.cuda.is_available():
        log.error("probe_budget_no_gpu")
        raise typer.Exit(1)

    lk = lake_root or LAKE_ROOT
    models = model_type or sorted(VALID_MODEL_TYPES)
    scales = scale or sorted(VALID_SCALES)
    convs: list[str | None] = conv_type if conv_type else [None]

    if dataset:
        datasets = list(dataset)
    else:
        datasets = [ds for ds in dataset_names()
                    if (cache_dir(lk, ds) / "cache_metadata.json").exists()]
    if not datasets:
        log.error("probe_budget_no_datasets", lake_root=lk)
        raise typer.Exit(1)

    combos = [(m, s, c, d) for m in models for s in scales for c in convs for d in datasets]
    log.info("probe_budget_start", combos=len(combos))

    if dry_run:
        for m, s, c, d in combos:
            log.info("probe_budget_plan", model=m, scale=s, conv=c or "default", dataset=d)
        raise typer.Exit()

    device = torch.device("cuda")

    with typer.progressbar(combos, label="probing", item_show_func=lambda c: c and f"{c[0]}/{c[1]}/{c[2] or 'default'}/{c[3]}") as bar:
        for mt, sc, ct, ds in bar:
            label = f"{mt}/{sc}/{ct or 'default'}/{ds}"
            try:
                from graphids.core.data.datasets.can_bus import CANBusDataset
                train_ds = CANBusDataset(root=cache_dir(lk, ds), raw_dir=data_dir(lk, ds), split="train")
                model = build_model_from_spec(
                    mt, sc, num_ids=train_ds.num_arb_ids,
                    in_channels=train_ds[0].x.shape[1], conv_type=ct,
                ).to(device)

                eff_conv = ct or getattr(model, "hparams", {}).get("conv_type", "gatv2")
                r = node_budget(ds, lk, model=model, train_dataset=train_ds, conv_type=eff_conv)

                attrs = {"model_type": mt, "scale": sc, "conv_type": eff_conv, "dataset": ds}
                _bpn_gauge.set(r.bytes_per_node or 0, attributes=attrs)
                _budget_gauge.set(r.budget, attributes=attrs)
                _bwd_gauge.set(r.backward_multiplier or 0, attributes=attrs)

                del model; torch.cuda.empty_cache()
                log.info("probe_done", label=label, budget=r.budget, bpn=r.bytes_per_node)
            except Exception as e:
                log.error("probe_failed", label=label, error=str(e))
