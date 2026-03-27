"""Experiment manifest: build configs, construct DAG, orchestrate SLURM execution."""
from __future__ import annotations

import dataclasses
import graphlib
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger()

_RESERVED_KEYS = frozenset({"stages"})


@dataclass(frozen=True)
class _StageJob:
    """One deduplicated SLURM job in the DAG."""

    node_id: str
    stage: str
    dataset: str
    seed: int
    overrides: tuple[str, ...]
    resources: dict[str, Any]
    dep_ids: tuple[str, ...]
    config_names: frozenset[str]


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------

def _identity_key(
    stage: str, dataset: str, seed: int,
    config: dict[str, Any], identity_keys: list[str],
) -> str:
    vals = "_".join(f"{k}={config.get(k, '_default_')}" for k in sorted(identity_keys))
    return f"{stage}|{dataset}|{seed}|{vals}" if vals else f"{stage}|{dataset}|{seed}"


def _dag_levels(dag: list[_StageJob]) -> list[list[_StageJob]]:
    depth: dict[str, int] = {}
    for job in dag:
        d = max((depth.get(dep, 0) for dep in job.dep_ids), default=-1) + 1
        depth[job.node_id] = d
    max_d = max(depth.values(), default=-1)
    return [[j for j in dag if depth[j.node_id] == d] for d in range(max_d + 1)]


def _sbatch(job: _StageJob, account: str) -> str:
    import subprocess

    from graphids.config import PROJECT_ROOT

    res = job.resources
    overrides = " ".join(job.overrides)
    partition = res["partition"]

    is_cpu = partition == "cpu"
    stage_args = f"--cache --dataset {job.dataset}" + (" --skip-tmpdir" if is_cpu else "")
    preamble_env = f'STAGE_DATA_ARGS="{stage_args}"'
    if is_cpu:
        preamble_env = f"SKIP_CUDA_CONF=1 {preamble_env}"

    wrap = f'{preamble_env} source scripts/slurm/_preamble.sh && python -m graphids {overrides}'
    cmd = [
        "sbatch", "--parsable", f"--chdir={PROJECT_ROOT}",
        f"--partition={partition}", f"--mem={res['memory_gb']}G",
        f"--time={res['walltime_min']}", f"--cpus-per-task={res['cpus']}",
        f"--account={account}", "--output=slurm_logs/%j.out",
        "--signal=B:USR1@300",
    ]
    if res.get("gpus"):
        cmd.append(f"--gres=gpu:{res['gpus']}")
    cmd.extend(["--wrap", wrap])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _poll(job_ids: list[str], interval: int = 60) -> dict[str, str]:
    import subprocess
    import time

    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
    while True:
        result = subprocess.run(
            ["sacct", "-j", ",".join(job_ids), "--format=JobIDRaw,State", "--noheader", "--parsable2"],
            capture_output=True, text=True,
        )
        states = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if parts[0] in job_ids:
                states[parts[0]] = parts[1].split()[0]

        if len(states) >= len(job_ids) and all(s in terminal for s in states.values()):
            return states
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class Manifest:
    """Experiment manifest: define configs → build DAG → write YAML or run on SLURM.

    Usage::

        m = Manifest(sweep={"dataset": ["set_01"], "seed": [42]},
                     defaults={"scale": "small", "training.loss_fn": "focal"})
        m.add("baseline")
        m.add("focal_loss", **{"training.loss_fn": "focal"})
        m.run(dry_run=True)
    """

    def __init__(
        self,
        sweep: dict[str, list[Any]],
        defaults: dict[str, Any],
        *,
        expand: dict[str, list[str]] | None = None,
        configs: dict[str, dict[str, Any]] | None = None,
    ):
        self.sweep = sweep
        self.defaults = dict(defaults)
        self._expand = expand or {}
        self._configs: dict[str, dict[str, Any]] = configs or {}

    # -- Config building ---------------------------------------------------

    def _apply_expand(self, overrides: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in overrides.items():
            if k in self._expand:
                for target in self._expand[k]:
                    result[target] = v
            else:
                result[k] = v
        return result

    def add(self, name: str, **overrides: Any) -> None:
        """Add a named config with only-override keys (diff from defaults)."""
        expanded = self._apply_expand(overrides)
        expanded_defaults = self._apply_expand(self.defaults)
        self._configs[name] = {
            k: v for k, v in expanded.items() if v != expanded_defaults.get(k)
        }

    def factorial(self, name_prefix: str, **axes: Any) -> None:
        """Add Cartesian product of parameter axes."""
        keys = list(axes.keys())
        values = [v if isinstance(v, (list, tuple)) else [v] for v in axes.values()]
        for combo in product(*values):
            varying = [
                str(v) for k, v in zip(keys, combo)
                if isinstance(axes[k], (list, tuple)) and len(axes[k]) > 1
            ] or [str(v) for v in combo]
            self.add(f"{name_prefix}_{'_'.join(varying)}", **dict(zip(keys, combo)))

    def sweep_axis(self, name_prefix: str, **overrides: Any) -> None:
        """Sweep over a single list-valued axis, keeping others fixed."""
        sweep_key = next(
            (k for k, v in overrides.items() if isinstance(v, (list, tuple))), None,
        )
        if sweep_key is None:
            self.add(name_prefix, **overrides)
            return
        sweep_vals = overrides.pop(sweep_key)
        for val in sweep_vals:
            self.add(f"{name_prefix}_{val}", **{sweep_key: val, **overrides})

    # -- Persistence -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> Manifest:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            sweep=raw.get("sweep", {}),
            defaults=raw.get("defaults", {}),
            configs=raw.get("configs", {}),
        )

    def write(self, path: str | Path) -> None:
        doc = {
            "sweep": self.sweep,
            "defaults": self._apply_expand(self.defaults),
            "configs": self._configs,
        }
        Path(path).write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))

    # -- DAG construction --------------------------------------------------

    def build_dag(self, *, filter_configs: list[str] | None = None) -> list[_StageJob]:
        """Expand sweep × configs → dedup → topo-sort → ordered job list."""
        from graphids.config import (
            CATALOG_PATH,
            DEFAULTS_DIR,
            PIPELINE_YAML,
            STAGE_DEPENDENCIES,
            STAGE_MODEL_MAP,
        )

        configs = self._configs
        if filter_configs:
            configs = {k: v for k, v in configs.items() if k in filter_configs}
        defaults = self._apply_expand(self.defaults)

        stages_def = PIPELINE_YAML["stages"]
        default_stages = PIPELINE_YAML.get("default_stages", list(stages_def.keys()))
        resources = yaml.safe_load((DEFAULTS_DIR / "resources.yaml").read_text())["resource_profiles"]

        valid_datasets = {k for k in yaml.safe_load(CATALOG_PATH.read_text()) if not k.startswith("_")}
        for ds in self.sweep.get("dataset", [defaults.get("dataset")]):
            if ds and ds not in valid_datasets:
                raise ValueError(f"Dataset '{ds}' not found. Available: {sorted(valid_datasets)}")

        sweep_points = [
            dict(zip(self.sweep.keys(), combo))
            for combo in product(*(self.sweep[k] for k in self.sweep))
        ]

        jobs: dict[str, _StageJob] = {}

        for config_name, overrides in configs.items():
            merged = {**defaults, **(overrides or {})}
            stages = list(merged.pop("stages", default_stages))

            for sweep_point in sweep_points:
                dataset = sweep_point.get("dataset", merged.get("dataset", "unknown"))
                seed = sweep_point.get("seed", merged.get("seed", 42))
                run_config = {**merged, **sweep_point}
                stage_node_ids: dict[str, str] = {}

                for stage in stages:
                    if stage not in stages_def:
                        log.warning("unknown_stage", stage=stage, config=config_name)
                        continue

                    sdef = stages_def[stage]
                    node_id = _identity_key(
                        stage, dataset, seed, run_config, sdef.get("identity_keys", []),
                    )

                    if node_id in jobs:
                        jobs[node_id] = dataclasses.replace(
                            jobs[node_id], config_names=jobs[node_id].config_names | {config_name},
                        )
                    else:
                        mode = sdef.get("mode", "gpu_train")
                        if mode not in resources:
                            raise ValueError(f"No resource profile '{mode}'. Check resources.yaml.")

                        dep_ids = [
                            stage_node_ids[dep_stage]
                            for _, dep_stage in STAGE_DEPENDENCIES.get(stage, [])
                            if dep_stage in stage_node_ids
                        ]
                        overrides_list = [
                            f"model_type={STAGE_MODEL_MAP[stage]}",
                            f"dataset={dataset}", f"seed={seed}", f"stage={stage}",
                        ] + [f"{k}={v}" for k, v in run_config.items() if k not in _RESERVED_KEYS]

                        jobs[node_id] = _StageJob(
                            node_id=node_id, stage=stage, dataset=dataset, seed=seed,
                            overrides=tuple(overrides_list), resources=resources[mode],
                            dep_ids=tuple(dep_ids), config_names=frozenset({config_name}),
                        )

                    stage_node_ids[stage] = node_id

        dep_graph = {j.node_id: set(j.dep_ids) for j in jobs.values()}
        order = list(graphlib.TopologicalSorter(dep_graph).static_order())
        return [jobs[nid] for nid in order if nid in jobs]

    # -- Execution ---------------------------------------------------------

    def summary(self, dag: list[_StageJob]) -> None:
        stage_counts = Counter(j.stage for j in dag)
        n_sweep = 1
        for vals in self.sweep.values():
            n_sweep *= len(vals)
        naive = len(self._configs) * n_sweep * len(stage_counts)
        log.info(
            "dag_plan", total_jobs=len(dag), naive_jobs=naive,
            savings=f"{naive - len(dag)} jobs saved by dedup",
            per_stage=dict(stage_counts),
        )

    def run(self, *, dry_run: bool = False, filter_configs: list[str] | None = None) -> None:
        """Build DAG and execute: dry-run prints plan, otherwise submits level by level."""
        from graphids.config import SLURM_ACCOUNT

        dag = self.build_dag(filter_configs=filter_configs)
        self.summary(dag)

        if dry_run:
            for job in dag:
                log.info("dry_run", node=job.node_id, stage=job.stage, configs=sorted(job.config_names))
            return

        for level in _dag_levels(dag):
            submitted = {}
            for job in level:
                jid = _sbatch(job, SLURM_ACCOUNT)
                submitted[jid] = job
                log.info("submitted", node=job.node_id, job_id=jid, stage=job.stage)

            if not submitted:
                continue

            states = _poll(list(submitted.keys()))
            failed = {jid: s for jid, s in states.items() if s != "COMPLETED"}
            if failed:
                for jid, state in failed.items():
                    log.error("job_failed", job_id=jid, state=state, node=submitted[jid].node_id)
                raise SystemExit(f"{len(failed)} jobs failed — aborting DAG")
            log.info("level_complete", jobs=len(submitted))
