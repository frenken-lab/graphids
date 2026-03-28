 Here's the concrete hit list — real API calls that replace real custom code:

  jsonargparse (already installed)

  ┌─────┬──────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┬────────────────┐
  │  #  │                             API                              │                                      Replaces                                      │     Impact     │
  ├─────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┼────────────────┤
  │ J1  │ Literal["vgae","gat","dgi"] type on Config.model_type        │ No validation exists — bad model_type silently produces wrong preset path          │ Bug prevention │
  ├─────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┼────────────────┤
  │ J2  │ ArgumentParser(env_prefix="KD_GAT", default_env=True)        │ 9 manual os.environ.get() calls in _compute_derived + module constants (~18 lines) │ -18 lines      │
  ├─────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┼────────────────┤
  │ J3  │ parser.parse_object(cfg_dict) at checkpoint reload           │ 12-line to_namespace() recursive converter                                         │ -12 lines      │
  ├─────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┼────────────────┤
  │ J4  │ Args: docstrings in dataclasses → auto --help                │ No help text on 70+ config fields                                                  │ UX fix, 0 code │
  ├─────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┼────────────────┤
  │ J5  │ ClosedUnitInterval for dropout/alpha, PositiveInt for epochs │ No numeric bounds — dropout=-0.5 passes silently                                   │ Bug prevention │
  └─────┴──────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┴────────────────┘

  Lightning (already installed)

  ┌─────┬─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────┐
  │  #  │                           API                           │                                                     Replaces                                                     │       Impact       │
  ├─────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ L1  │ SLURMEnvironment(auto_requeue=True) as Trainer plugin   │ SLURM auto-save is not wired up — _resume_ckpt_path checks for .pl_auto_save.ckpt that's never written (~10      │ Bug fix + -10      │
  │     │                                                         │ lines dead)                                                                                                      │ lines              │
  ├─────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ L2  │ Reuse one Trainer across test_model /                   │ Creates N+1 throwaway Trainers per eval — one per test scenario                                                  │ -15 lines          │
  │     │ eval_with_scenarios                                     │                                                                                                                  │                    │
  ├─────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ L3  │ seed_everything(seed, workers=True)                     │ Missing workers=True — DataLoader workers not reproducibly seeded                                                │ Bug fix, 2 chars   │
  ├─────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ L4  │ configure_optimizers dict return with lr_scheduler      │ No LR scheduler at all — SWA callback is the only LR manipulation                                                │ Enables feature    │
  ├─────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ L5  │ GradientAccumulationScheduler callback                  │ Static accumulate_grad_batches int                                                                               │ Enables feature
