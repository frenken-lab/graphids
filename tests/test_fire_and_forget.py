"""Tests for fire_and_forget() zero-daemon SLURM submission.

Monkeypatches submit_no_poll to capture calls without touching SLURM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class _SubmitCall:
    """Captured call to submit_no_poll."""
    stage: str
    model: str
    scale: str
    dataset: str
    seed: int | None
    auxiliaries: str
    dependency_job_id: str | None


@dataclass
class _SubmitRecorder:
    """Records all calls to the fake submit_no_poll."""
    calls: list[_SubmitCall] = field(default_factory=list)
    counter: int = 0

    def __call__(
        self,
        stage, model, scale, dataset, resources,
        *, seed=None, auxiliaries="none", dependency_job_id=None, dry_run=False,
    ) -> str:
        self.calls.append(_SubmitCall(
            stage=stage, model=model, scale=scale, dataset=dataset,
            seed=seed, auxiliaries=auxiliaries, dependency_job_id=dependency_job_id,
        ))
        self.counter += 1
        return f"job-{self.counter}"


@pytest.fixture()
def recorder(monkeypatch):
    """Monkeypatch submit_no_poll and return the call recorder."""
    from graphids.pipeline.orchestration import dagster_defs

    rec = _SubmitRecorder()
    monkeypatch.setattr(dagster_defs, "submit_no_poll", rec)
    return rec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFireAndForget:
    def test_submits_all_stages(self, recorder):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        job_ids = fire_and_forget(dataset="hcrl_sa", dry_run=False)
        # 13 DAG nodes (preprocess + 4×3 variants), single seed
        assert len(job_ids) == 13
        assert len(recorder.calls) == 13

    def test_all_job_ids_unique(self, recorder):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        job_ids = fire_and_forget(dataset="hcrl_sa")
        assert len(set(job_ids.values())) == len(job_ids)

    def test_preprocess_submitted_first(self, recorder):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        fire_and_forget(dataset="hcrl_sa")
        assert recorder.calls[0].stage == "preprocess"

    def test_preprocess_has_no_dependency(self, recorder):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        fire_and_forget(dataset="hcrl_sa")
        assert recorder.calls[0].dependency_job_id is None

    def test_dependent_stages_get_parent_job_ids(self, recorder):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        fire_and_forget(dataset="hcrl_sa")
        # Find autoencoder calls — they depend on preprocess
        ae_calls = [c for c in recorder.calls if c.stage == "autoencoder"]
        for call in ae_calls:
            # Should have a dependency_job_id (the preprocess job)
            assert call.dependency_job_id is not None, (
                f"autoencoder {call.model}_{call.scale} has no dependency"
            )

    def test_multi_seed_multiplies_jobs(self, recorder):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        job_ids = fire_and_forget(dataset="hcrl_sa", seeds=[42, 123])
        assert len(job_ids) == 26  # 13 × 2
        assert len(recorder.calls) == 26

    def test_multi_seed_keys_have_seed_suffix(self, recorder):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        job_ids = fire_and_forget(dataset="hcrl_sa", seeds=[42, 123])
        assert any("seed42" in k for k in job_ids)
        assert any("seed123" in k for k in job_ids)

    def test_dataset_forwarded_to_all_calls(self, recorder):
        from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

        fire_and_forget(dataset="hcrl_ch")
        for call in recorder.calls:
            assert call.dataset == "hcrl_ch"
