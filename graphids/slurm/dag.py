"""OFAT ablation DAG — declarative topology + topological submission.

The DAG is expressed as a tuple of :class:`FitNode`\\ s plus one
:class:`ExtractStatesNode`. ``execute_dag`` walks it in topological
order, submits each node via :func:`graphids.slurm.submit.submit`,
and chains an afterok test job for each fit. Dependencies are passed
as ``dep_jids`` ints held in memory — no scheduler re-query between
stages (kills the Stage 3 dep-race). See PLAN.md "Phase A simplification
(executed 2026-04-23)".

Current shape is declarative data (OFAT_DAG) consumed by an imperative
executor. Future evolution: load the DAG from a jsonnet file so the
topology becomes inspectable (``jsonnet configs/ablation_dag.jsonnet``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from graphids.slurm.submit import load_dotenv, submit

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 777)
DEFAULT_DATASET: str = "set_01"

# Groups whose presets exceed the profile's default `long` wall.
# Empirical (2026-04-23, Cardinal H100): VGAE/GAE/DGI at max_epochs=1200,
# ~9 s/epoch → ~180 min needed. Profile default is 90 min on Cardinal.
LONG_TIMEOUT_MIN: int = 210

# Sentinel for "fit already FINISHED per MLflow; no new jid, still chain test".
_FIT_ALREADY_COMPLETE: int = -1

# Name of the extract-fusion-states node (distinct from any preset variant).
EXTRACT_STATES_NAME: str = "extract_states"


@dataclass(frozen=True)
class FitNode:
    """One preset-based fit with an afterok test chained behind it.

    ``deps`` names other nodes in the same DAG; their jids become
    ``afterok`` deps on this node's fit submission.
    """

    name: str
    preset_path: str  # relative to configs/
    group: str
    variant: str
    deps: tuple[str, ...] = ()
    timeout_min: int | None = None  # None → use profile default


@dataclass(frozen=True)
class ExtractStatesNode:
    """Fusion-state extraction. Only non-preset job in the DAG.

    Hard-coded shape (upstream = vgae + focal, output = states file) —
    if we ever add a second non-preset node, generalize this then.
    """

    name: str = EXTRACT_STATES_NAME
    deps: tuple[str, ...] = ("vgae", "focal")
    timeout_min: int = 30
    mem_gb: int = 36


DagNode = FitNode | ExtractStatesNode


@dataclass
class DagResult:
    """Return value of :func:`execute_dag`."""

    # (node_name, seed) → jid. _FIT_ALREADY_COMPLETE for idempotent skips.
    jids: dict[tuple[str, int], int] = field(default_factory=dict)
    skipped: list[tuple[str, int]] = field(default_factory=list)


# --------------------------------------------------------------------------
# The OFAT DAG — declarative topology.
# --------------------------------------------------------------------------
#
# Stage boundaries are implicit in the deps. `(extract_states,)` fans 4
# fusion variants in; `(vgae,)` gates curriculum_vgae. Everything else is
# a free standalone fit.

OFAT_DAG: tuple[DagNode, ...] = (
    # Stage 0 — unsupervised baselines (VGAE jid needed downstream).
    FitNode(
        "vgae", "unsupervised/vgae.jsonnet", "unsupervised", "vgae", timeout_min=LONG_TIMEOUT_MIN
    ),
    FitNode("gae", "unsupervised/gae.jsonnet", "unsupervised", "gae", timeout_min=LONG_TIMEOUT_MIN),
    FitNode("dgi", "unsupervised/dgi.jsonnet", "unsupervised", "dgi", timeout_min=LONG_TIMEOUT_MIN),
    # Stage 1 — standalone parallel variants.
    FitNode("gat", "conv_type/gat.jsonnet", "conv_type", "gat"),
    FitNode("gatv2", "conv_type/gatv2.jsonnet", "conv_type", "gatv2"),
    FitNode("gps", "conv_type/gps.jsonnet", "conv_type", "gps"),
    FitNode("none", "gat_sampling/none.jsonnet", "gat_sampling", "none"),
    FitNode(
        "curriculum_random",
        "gat_sampling/curriculum_random.jsonnet",
        "gat_sampling",
        "curriculum_random",
    ),
    FitNode("ce", "gat_loss/ce.jsonnet", "gat_loss", "ce"),
    FitNode("weighted_ce", "gat_loss/weighted_ce.jsonnet", "gat_loss", "weighted_ce"),
    # focal jid is needed downstream by extract_states.
    FitNode("focal", "gat_loss/focal.jsonnet", "gat_loss", "focal"),
    FitNode("lookup", "id_encoding/lookup.jsonnet", "id_encoding", "lookup"),
    FitNode("learned_unk", "id_encoding/learned_unk.jsonnet", "id_encoding", "learned_unk"),
    FitNode("hash", "id_encoding/hash.jsonnet", "id_encoding", "hash"),
    # Stage 2 — curriculum_vgae fans in to vgae.
    FitNode(
        "curriculum_vgae",
        "gat_sampling/curriculum_vgae.jsonnet",
        "gat_sampling",
        "curriculum_vgae",
        deps=("vgae",),
    ),
    # Stage 3 — extract fusion states (vgae + focal encoders → tensor cache).
    ExtractStatesNode(),
    # Stage 4 — fusion methods fan out from extract_states.
    FitNode("bandit", "fusion/bandit.jsonnet", "fusion", "bandit", deps=(EXTRACT_STATES_NAME,)),
    FitNode("dqn", "fusion/dqn.jsonnet", "fusion", "dqn", deps=(EXTRACT_STATES_NAME,)),
    FitNode("mlp", "fusion/mlp.jsonnet", "fusion", "mlp", deps=(EXTRACT_STATES_NAME,)),
    FitNode(
        "weighted_avg",
        "fusion/weighted_avg.jsonnet",
        "fusion",
        "weighted_avg",
        deps=(EXTRACT_STATES_NAME,),
    ),
)


# --------------------------------------------------------------------------
# MLflow idempotency.
# --------------------------------------------------------------------------


def fit_is_complete(dataset: str, group: str, variant: str, seed: int) -> bool:
    """Return True iff the *latest* fit attempt for this variant+seed is FINISHED.

    "Latest" matters: MLflow accumulates history across refactors, so an
    old FINISHED run from a prior code version can coexist with today's
    FAILED/RUNNING run. We only trust the most recent attempt. RUNNING
    (incl. zombies from SLURM-killed processes whose MLflow row never
    flipped) returns False → safe re-run.
    """
    import mlflow

    from graphids._mlflow import build_search_filter, ensure_tracking_uri

    uri = ensure_tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)

    df = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=build_search_filter(
            dataset=dataset, group=group, variant=variant, seed=seed, phase="fit"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return not df.empty and df.iloc[0]["status"] == "FINISHED"


# --------------------------------------------------------------------------
# Submission helpers — direct calls into ``graphids.slurm.submit.submit``.
# --------------------------------------------------------------------------


def _run_dir(lake_root: Path, dataset: str, group: str, variant: str, seed: int) -> Path:
    return lake_root / dataset / "ablations" / group / variant / f"seed_{seed}"


def submit_fit(
    node: FitNode,
    seed: int,
    *,
    dataset: str,
    lake_root: Path,
    cluster: str,
    configs_root: Path,
    dep_jids: Sequence[int] = (),
    dry_run: bool = False,
) -> int:
    """Submit a fit job. Returns jid (>0), or ``_FIT_ALREADY_COMPLETE``.

    Dry-run returns :data:`graphids.slurm.submit.DRY_RUN_JID` (0) as a
    non-chaining sentinel.
    """
    if fit_is_complete(dataset, node.group, node.variant, seed):
        print(
            f"  [skip] fit already FINISHED in MLflow: {node.group}/{node.variant} seed_{seed}",
            file=sys.stderr,
        )
        return _FIT_ALREADY_COMPLETE

    return submit(
        preset=configs_root / node.preset_path,
        tlas=[("dataset", dataset), ("seed", seed), ("lake_root", str(lake_root))],
        cluster=cluster or None,
        timeout_min=node.timeout_min,
        dep_jids=dep_jids,
        dry_run=dry_run,
    )


def chain_test(
    node: FitNode,
    seed: int,
    *,
    dataset: str,
    lake_root: Path,
    cluster: str,
    configs_root: Path,
    fit_jid: int,
    dry_run: bool = False,
) -> None:
    """Submit afterok CPU test for a fit node.

    Runs without dep if fit was already complete (``_FIT_ALREADY_COMPLETE``)
    — test fires immediately against the existing ckpt.

    ``mem_gb=32`` because ``classification_test_metrics`` buffers full
    (N, K) probs on CPU; the 16G profile default OOM'd on seed 42 (test-gat
    8724434 at 16.77G). Long-term fix is stream-compute via torchmetrics.
    """
    if fit_jid == 0:  # dry-run sentinel — don't chain
        return

    ckpt = (
        _run_dir(lake_root, dataset, node.group, node.variant, seed)
        / "checkpoints"
        / "best_model.ckpt"
    )
    submit(
        preset=configs_root / node.preset_path,
        action="test",
        mode="cpu",
        length="short",
        mem_gb=32,
        tlas=[("dataset", dataset), ("seed", seed), ("lake_root", str(lake_root))],
        ckpt_path=str(ckpt),
        cluster=cluster or None,
        dep_jids=(fit_jid,) if fit_jid > 0 else (),
        dry_run=dry_run,
    )


def submit_extract_states(
    node: ExtractStatesNode,
    seed: int,
    *,
    dataset: str,
    lake_root: Path,
    cluster: str,
    dep_jids: Sequence[int] = (),
    dry_run: bool = False,
) -> int:
    """Submit fusion-state extraction. Afterok (vgae, focal) by convention."""
    vgae_ckpt = (
        _run_dir(lake_root, dataset, "unsupervised", "vgae", seed)
        / "checkpoints"
        / "best_model.ckpt"
    )
    gat_ckpt = (
        _run_dir(lake_root, dataset, "gat_loss", "focal", seed) / "checkpoints" / "best_model.ckpt"
    )
    out = lake_root / dataset / "ablations" / "fusion_states" / f"seed_{seed}"
    inner = (
        f"python -m graphids extract-fusion-states "
        f"--vgae-ckpt {vgae_ckpt} --gat-ckpt {gat_ckpt} "
        f"--dataset {dataset} --seed {seed} --output-dir {out}"
    )
    return submit(
        command=inner,
        mode="gpu",
        mem_gb=node.mem_gb,
        timeout_min=node.timeout_min,
        cluster=cluster or None,
        dep_jids=dep_jids,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------
# DAG executor.
# --------------------------------------------------------------------------


def _toposort(nodes: Iterable[DagNode]) -> list[DagNode]:
    """Stable topological sort by dep-names. Raises on cycle or missing dep."""
    from graphlib import TopologicalSorter

    by_name = {n.name: n for n in nodes}
    for n in by_name.values():
        for dep in n.deps:
            if dep not in by_name:
                raise RuntimeError(f"node {n.name!r} deps on unknown {dep!r}")
    ts = TopologicalSorter({n.name: set(n.deps) for n in by_name.values()})
    return [by_name[name] for name in ts.static_order()]


def execute_dag(
    nodes: Iterable[DagNode],
    *,
    dataset: str,
    seeds: Sequence[int],
    cluster: str,
    lake_root: Path,
    configs_root: Path,
    dry_run: bool = False,
) -> DagResult:
    """Submit every (node, seed) pair in topological order.

    For each node and each seed, we look up the seed-scoped jids of its
    deps from the running jid map and pass them as afterok deps. Fit
    nodes get a test job chained; ExtractStates is a single submission.
    """
    ordered = _toposort(nodes)
    result = DagResult()

    for n in ordered:
        header = f"=== {n.name} ({len(seeds)} seeds"
        if n.deps:
            header += f", deps={list(n.deps)}"
        print(header + ") ===", file=sys.stderr)

        for seed in seeds:
            dep_jids = [result.jids[(d, seed)] for d in n.deps]
            if isinstance(n, FitNode):
                jid = submit_fit(
                    n,
                    seed,
                    dataset=dataset,
                    lake_root=lake_root,
                    cluster=cluster,
                    configs_root=configs_root,
                    dep_jids=dep_jids,
                    dry_run=dry_run,
                )
                result.jids[(n.name, seed)] = jid
                if jid == _FIT_ALREADY_COMPLETE:
                    result.skipped.append((n.name, seed))
                chain_test(
                    n,
                    seed,
                    dataset=dataset,
                    lake_root=lake_root,
                    cluster=cluster,
                    configs_root=configs_root,
                    fit_jid=jid,
                    dry_run=dry_run,
                )
            elif isinstance(n, ExtractStatesNode):
                jid = submit_extract_states(
                    n,
                    seed,
                    dataset=dataset,
                    lake_root=lake_root,
                    cluster=cluster,
                    dep_jids=dep_jids,
                    dry_run=dry_run,
                )
                result.jids[(n.name, seed)] = jid
            print(f"  seed={seed} -> {n.name} jid={result.jids[(n.name, seed)]}", file=sys.stderr)

    return result


# --------------------------------------------------------------------------
# Top-level entrypoint used by the CLI.
# --------------------------------------------------------------------------


def launch_ablation(
    *,
    dataset: str = DEFAULT_DATASET,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cluster: str = "",
    dry_run: bool = False,
    project_root: Path | None = None,
) -> DagResult:
    """Submit the full OFAT DAG for ``seeds`` against ``dataset``.

    Sources ``.env`` for ``GRAPHIDS_LAKE_ROOT`` / ``USER``; submit() itself
    is CWD-agnostic (uses absolute paths throughout).
    """
    project_root = project_root or Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    lake_env = os.environ.get("GRAPHIDS_LAKE_ROOT")
    if not lake_env:
        raise RuntimeError("GRAPHIDS_LAKE_ROOT must be set in .env")
    lake_root = Path(lake_env) / "dev" / os.environ["USER"]

    return execute_dag(
        OFAT_DAG,
        dataset=dataset,
        seeds=seeds,
        cluster=cluster,
        lake_root=lake_root,
        configs_root=project_root / "configs" / "ablations",
        dry_run=dry_run,
    )
