"""Typer CLI app for GraphIDS.

No torch / model imports at module level — safe on login nodes.
Heavy imports are deferred to inside command functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    name="graphids",
    help="GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# Root callback — runs once per CLI invocation before any subcommand.
# Scoped to cheap setup only (logging level + OTel providers). ensure_spawn()
# imports torch and so stays inside command bodies so ``<cmd> --help`` keeps
# its fast path on login nodes.
# ---------------------------------------------------------------------------


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Debug-level logging on the graphids logger"),
    ] = False,
) -> None:
    """GraphIDS CLI — shared setup for every subcommand."""
    import logging
    import os

    from graphids._otel import init_providers

    logging.getLogger("graphids").setLevel(logging.DEBUG if verbose else logging.INFO)
    init_providers(
        "graphids",
        wandb_entity=os.environ.get("WANDB_ENTITY", ""),
        wandb_project=os.environ.get("WANDB_PROJECT", "graphids"),
    )
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# Per-element parsers (Typer option `parser=` callbacks)
# ---------------------------------------------------------------------------


def _parse_kv_pair(raw: str) -> tuple[str, Any]:
    """Parse one ``key=value`` flag into a typed ``(key, value)`` pair.

    JSON-decodes the value; bare unquoted identifiers fall through as strings.
    Shared by ``--tla`` (key is a jsonnet TLA name) and ``--set`` (key is a
    dotted path into the rendered dict).
    """
    key, eq, val = raw.partition("=")
    if not eq:
        raise typer.BadParameter(f"expected key=value, got {raw!r}")
    try:
        return key, json.loads(val)
    except json.JSONDecodeError:
        return key, val


# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

ConfigPath = Annotated[
    Path,
    typer.Option(
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to jsonnet stage config",
    ),
]
# The ``parser=`` callback returns ``(key, value)`` tuples at runtime, but
# Typer's annotation inspector can't handle ``list[tuple[...]]`` — so the
# annotation lies with ``list[str] | None`` and consumers do ``dict(tla or [])``
# to recover the mapping. This keeps validation + metavar inside the Option
# decl while staying within what Typer's type system supports.
TlaList = Annotated[
    list[str] | None,
    typer.Option(
        "--tla",
        parser=_parse_kv_pair,
        metavar="KEY=JSON",
        help="key=value TLA for jsonnet (repeatable)",
    ),
]
SetList = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        parser=_parse_kv_pair,
        metavar="DOTTED.PATH=JSON",
        help="dotted.path=value override on rendered dict (repeatable)",
    ),
]
CkptPath = Annotated[
    str | None, typer.Option("--ckpt-path", help="Checkpoint path for trainer method")
]


# ---------------------------------------------------------------------------
# Shell completion helpers
# ---------------------------------------------------------------------------
#
# Each ``_complete_*`` takes an ``incomplete`` prefix (typer passes whatever the
# user has typed so far after ``<TAB>``) and returns the matching values. These
# are wired via ``autocompletion=`` on the relevant Options in ``pipeline.py``
# and ``data.py``. Values come from the authoritative source
# (topology catalog, pydantic Literal, axes.json frozenset) — no hardcoded lists
# that can drift. Each helper defers its imports so ``--help`` stays fast.


def _complete_dataset(incomplete: str) -> list[str]:
    from graphids.config.topology import dataset_names

    return [n for n in dataset_names() if n.startswith(incomplete)]


def _complete_scale(incomplete: str) -> list[str]:
    from graphids.config.constants import VALID_SCALES

    return sorted(v for v in VALID_SCALES if v.startswith(incomplete))


def _complete_fusion_method(incomplete: str) -> list[str]:
    from graphids.config.constants import VALID_FUSION_METHODS

    return sorted(v for v in VALID_FUSION_METHODS if v.startswith(incomplete))


def _complete_model_type(incomplete: str) -> list[str]:
    from graphids.config.constants import VALID_MODEL_TYPES

    return sorted(v for v in VALID_MODEL_TYPES if v.startswith(incomplete))


def _literal_field_values(field_name: str) -> tuple[str, ...]:
    """Extract allowed values from a ``PipelineConfig`` Literal-typed field.

    Single source of truth for conv_type / loss_fn completion: the pydantic
    model owns the Literal, we just read its arguments via ``typing.get_args``.
    """
    from typing import get_args

    from graphids.orchestrate.config import PipelineConfig

    return get_args(PipelineConfig.model_fields[field_name].annotation)


def _complete_conv_type(incomplete: str) -> list[str]:
    return [v for v in _literal_field_values("conv_type") if v.startswith(incomplete)]


def _complete_loss_fn(incomplete: str) -> list[str]:
    return [v for v in _literal_field_values("loss_fn") if v.startswith(incomplete)]


def apply_overrides(
    rendered: dict[str, Any],
    overrides: list[tuple[str, Any]] | None,
) -> None:
    """Apply pre-parsed ``dotted.path=value`` overrides in-place on a rendered dict."""
    for key, typed_val in overrides or []:
        parts = key.split(".")
        cur: Any = rendered
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = typed_val


# ---------------------------------------------------------------------------
# Meta commands — operational lookups that don't fit any domain panel
# ---------------------------------------------------------------------------


@app.command("submit-profile", rich_help_panel="Meta")
def submit_profile(
    job: str,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            help="Size mem/time from this dataset's cache_metadata.json",
            autocompletion=_complete_dataset,
        ),
    ] = None,
    scale: Annotated[
        str,
        typer.Option(
            "--scale", help="Model scale for per-stage scaling", autocompletion=_complete_scale
        ),
    ] = "small",
) -> None:
    """Print resource profile fields for ``scripts/slurm/submit.sh``.

    Reads ``configs/resources/submit_profiles.json`` and prints
    ``partition cpus mem time signal mode gres command`` on a single line.

    Profiles fall into three shapes:
    - static (``time`` + ``mem`` fields present) — emit as-is
    - scaling (``scaling`` block) — size from ``--dataset``'s num_raw_samples;
      fall back to ``defaults`` when ``--dataset`` is omitted
    - composed (``stages`` list) — per-stage sizing then compose:
      ``time = sum(stages)``, ``cpus/mem = max(stages)``
    """
    from graphids._otel import get_logger
    from graphids.config.constants import PROJECT_ROOT

    log = get_logger(__name__)
    config = json.loads(
        (PROJECT_ROOT / "configs" / "resources" / "submit_profiles.json").read_text()
    )
    submit_profiles = config["submit_profiles"]
    stage_profiles = config.get("stage_profiles", {})
    if job not in submit_profiles:
        log.error("submit_profile_unknown", job=job, available=sorted(submit_profiles))
        raise typer.Exit(1)
    p = submit_profiles[job]

    num_raw = _load_num_raw_samples(dataset) if dataset else None
    cpus, mem_str, time_str = _resolve_resources(p, stage_profiles, num_raw, scale)

    gres = "gpu:1" if p["mode"] == "gpu" else "NONE"
    signal = p.get("signal") or "NONE"
    print(
        f"{p['partition']} {cpus} {mem_str} {time_str} {signal} {p['mode']} {gres} {p['command']}"
    )


def _load_num_raw_samples(dataset: str) -> int | None:
    """Load ``aggregate.num_raw_samples`` from the dataset's cache metadata."""
    from graphids.config.constants import LAKE_ROOT
    from graphids.config.topology import cache_dir
    from graphids.core.data.metadata import load_metadata

    try:
        meta = load_metadata(cache_dir(LAKE_ROOT, dataset))
    except (FileNotFoundError, ValueError):
        return None
    return int(meta.get("aggregate", {}).get("num_raw_samples") or 0) or None


def _eval_scaling(block: dict[str, Any], num_raw: int | None, scale: str) -> float:
    """Compute ``base + per_mraw * mraw``, multiplied by scale factor if present."""
    mraw = (num_raw or 0) / 1e6
    value = block["base"] + block.get("per_mraw", 0.0) * mraw
    mults = block.get("scale_mult") or {}
    return float(value * mults.get(scale, 1.0))


def _size_from_scaling(
    profile: dict[str, Any], num_raw: int | None, scale: str
) -> tuple[int, float, float]:
    """Return ``(cpus, mem_gb, time_min)`` for a profile with a ``scaling`` block."""
    sc = profile["scaling"]
    return (
        int(profile["cpus"]),
        _eval_scaling(sc["mem_gb"], num_raw, scale),
        _eval_scaling(sc["time_min"], num_raw, scale),
    )


def _resolve_resources(
    profile: dict[str, Any],
    stage_profiles: dict[str, Any],
    num_raw: int | None,
    scale: str,
) -> tuple[int, str, str]:
    """Resolve (cpus, mem, time) for any profile shape. Returns sbatch-ready strings."""
    import math

    if "stages" in profile:
        if num_raw is None:
            d = profile["defaults"]
            return int(d["cpus"]), str(d["mem"]), str(d["time"])
        per_stage = [
            _size_from_scaling(stage_profiles[s], num_raw, scale) for s in profile["stages"]
        ]
        cpus = max(c for c, _, _ in per_stage)
        mem_gb = max(m for _, m, _ in per_stage)
        time_min = sum(t for _, _, t in per_stage)
    elif "scaling" in profile:
        if num_raw is None:
            d = profile["defaults"]
            return int(profile["cpus"]), str(d["mem"]), str(d["time"])
        cpus, mem_gb, time_min = _size_from_scaling(profile, num_raw, scale)
    else:
        return int(profile["cpus"]), str(profile["mem"]), str(profile["time"])

    return cpus, f"{math.ceil(mem_gb)}G", _format_time(time_min)


def _format_time(minutes: float) -> str:
    """Convert float minutes to ``H:MM:SS`` (ceil to whole minute)."""
    import math

    total_min = max(1, math.ceil(minutes))
    h, m = divmod(total_min, 60)
    return f"{h}:{m:02d}:00"
