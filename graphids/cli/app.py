"""Typer CLI app for GraphIDS.

No torch / model imports at module level — safe on login nodes.
Heavy imports are deferred to inside command functions.
"""

from __future__ import annotations

import datetime
import json
import statistics
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
# user has typed so far after ``<TAB>``) and returns the matching values. Values
# come from the authoritative source (dataset catalog, axes.json frozenset) — no
# hardcoded lists that can drift. Each helper defers its imports so ``--help``
# stays fast.


def _complete_dataset(incomplete: str) -> list[str]:
    from graphids.config.topology import dataset_names

    return [n for n in dataset_names() if n.startswith(incomplete)]


def _complete_scale(incomplete: str) -> list[str]:
    from graphids.config.constants import VALID_SCALES

    return sorted(v for v in VALID_SCALES if v.startswith(incomplete))


def _complete_model_type(incomplete: str) -> list[str]:
    from graphids.config.constants import VALID_MODEL_TYPES

    return sorted(v for v in VALID_MODEL_TYPES if v.startswith(incomplete))


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
    cluster: Annotated[
        str,
        typer.Option("--cluster", help="Target cluster (picks partition for fit-shape profiles)"),
    ] = "pitzer",
    length: Annotated[
        str,
        typer.Option("--length", help="short (debug queue) or long (batch queue)"),
    ] = "long",
    group: Annotated[
        str | None,
        typer.Option(
            "--group",
            help="Ablation group (conv_type / gat_loss / ...) — enables MLflow history lookup",
        ),
    ] = None,
) -> None:
    """Print resource profile fields for ``scripts/slurm/submit.sh``.

    Reads ``configs/resources/submit_profiles.json`` and prints
    ``partition cpus mem time signal mode gres command`` on a single line.

    Profiles fall into three shapes:
    - static (``partition`` + ``time`` + ``mem`` fields) — emit as-is.
    - scaling (``scaling`` block) — size from ``--dataset``'s num_raw_samples;
      fall back to ``defaults`` when ``--dataset`` is omitted.
    - fit-shape (``partitions`` + ``times`` per-cluster maps) — pick partition
      from ``partitions[cluster][length]`` and time from MLflow history for
      matching ``(cluster, group, dataset)`` runs (p95 × 1.5 buffer) if
      available; fall back to ``times[cluster][length]`` otherwise.
    """
    from graphids._otel import get_logger
    from graphids.config.constants import PROJECT_ROOT

    log = get_logger(__name__)
    config = json.loads(
        (PROJECT_ROOT / "configs" / "resources" / "submit_profiles.json").read_text()
    )
    submit_profiles = config["submit_profiles"]
    if job not in submit_profiles:
        log.error("submit_profile_unknown", job=job, available=sorted(submit_profiles))
        raise typer.Exit(1)
    p = dict(submit_profiles[job])

    if "partitions" in p:
        cluster_map = p["partitions"].get(cluster)
        if cluster_map is None:
            log.error(
                "submit_profile_unknown_cluster",
                cluster=cluster,
                available=sorted(p["partitions"]),
            )
            raise typer.Exit(1)
        if length not in cluster_map:
            log.error("submit_profile_unknown_length", length=length, available=sorted(cluster_map))
            raise typer.Exit(1)
        p["partition"] = cluster_map[length]
        # Time resolution: try MLflow history first, then per-cluster static.
        p["time"] = _resolve_fit_time(p, cluster, length, group, dataset, log)

    num_raw = _load_num_raw_samples(dataset) if dataset else None
    cpus, mem_str, time_str = _resolve_resources(p, num_raw, scale)

    gres = "gpu:1" if p["mode"] == "gpu" else "NONE"
    signal = p.get("signal") or "NONE"
    print(
        f"{p['partition']} {cpus} {mem_str} {time_str} {signal} {p['mode']} {gres} {p['command']}"
    )


def _resolve_fit_time(
    p: dict,
    cluster: str,
    length: str,
    group: str | None,
    dataset: str | None,
    log,
) -> str:
    """Resolve walltime for a fit-shape profile. History-first, static fallback.

    Query MLflow for prior FINISHED fit-phase runs matching
    ``(cluster, group, dataset)``. If ≥3 runs exist, use ``p95 × 1.5`` as the
    walltime (bounded by the per-cluster long/short static as a ceiling for
    ``short`` length to keep smoke tests inside the debug partition limit).
    Otherwise fall back to ``times[cluster][length]`` — a hand-calibrated
    per-cluster static that acknowledges GPU-throughput differences.
    """
    fallback = p["times"][cluster][length]
    if not group or not dataset:
        return fallback
    if length == "short":
        # Short runs are smoke tests capped by debug-queue wall limits; never
        # trust history (a slow "long" run would push short past the cap).
        return fallback
    mins = _estimate_walltime_minutes(cluster, group, dataset)
    if mins is None:
        return fallback
    log.info(
        "walltime_from_history",
        cluster=cluster,
        group=group,
        dataset=dataset,
        minutes=mins,
        fallback=fallback,
    )
    # sbatch accepts HH:MM:SS up to 7 days; format explicitly from total_seconds
    # since timedelta's default str emits "D days, H:MM:SS" when days>0.
    td = datetime.timedelta(minutes=mins)
    total = int(td.total_seconds())
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:00"


def _estimate_walltime_minutes(cluster: str, group: str, dataset: str) -> int | None:
    """Query MLflow for prior ``(cluster, group, dataset)`` FINISHED fit runs.

    Returns ``ceil(p95(elapsed_mins) * 1.5)``, clamped to ``[10, 10080]``
    (10 min floor, 7-day SLURM ceiling). ``None`` if fewer than 3 matching
    runs exist — the caller falls back to the per-cluster static.
    """
    import math

    try:
        from graphids._mlflow import ensure_tracking_uri
    except ImportError:
        return None

    uri = ensure_tracking_uri()
    if uri is None:
        return None
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None
    try:
        client = MlflowClient(tracking_uri=uri)
        experiments = [e.experiment_id for e in client.search_experiments()]
        if not experiments:
            return None
        # Use slurm.slurm_cluster_name (set by SLURM itself, always correct)
        # over graphids.cluster (derived from settings.cluster, can be empty
        # when the submitter shell's GRAPHIDS_CLUSTER isn't exported into the
        # job env). Same source-of-truth reasoning as the unsupervised runs'
        # backfill script.
        filter_str = (
            f"tags.`slurm.slurm_cluster_name` = '{cluster}' "
            f"AND tags.`graphids.group` = '{group}' "
            f"AND tags.`graphids.dataset` = '{dataset}' "
            f"AND tags.`graphids.phase` = 'fit' "
            f"AND attributes.status = 'FINISHED'"
        )
        runs = client.search_runs(
            experiment_ids=experiments, filter_string=filter_str, max_results=50
        )
    except Exception:
        return None
    elapsed = [
        (r.info.end_time - r.info.start_time) / 60000
        for r in runs
        if r.info.end_time and r.info.start_time and r.info.end_time > r.info.start_time
    ]
    if len(elapsed) < 3:
        return None
    # statistics.quantiles requires n>=2; the len(elapsed) < 3 guard above covers it.
    p95 = statistics.quantiles(elapsed, n=100, method="inclusive")[94]
    return max(10, min(int(math.ceil(p95 * 1.5)), 7 * 24 * 60))


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
    num_raw: int | None,
    scale: str,
) -> tuple[int, str, str]:
    """Resolve (cpus, mem, time) for a profile. Returns sbatch-ready strings."""
    import math

    if "scaling" in profile:
        if num_raw is None:
            d = profile["defaults"]
            return int(profile["cpus"]), str(d["mem"]), str(d["time"])
        cpus, mem_gb, time_min = _size_from_scaling(profile, num_raw, scale)
        return cpus, f"{math.ceil(mem_gb)}G", _format_time(time_min)
    return int(profile["cpus"]), str(profile["mem"]), str(profile["time"])


def _format_time(minutes: float) -> str:
    """Convert float minutes to ``H:MM:SS`` (ceil to whole minute)."""
    import math

    total_min = max(1, math.ceil(minutes))
    h, m = divmod(total_min, 60)
    return f"{h}:{m:02d}:00"
