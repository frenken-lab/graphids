Thorough results. Here's the distilled version.

---

## 3 Config Design Patterns for Taming Combinatorial Explosion

### Pattern 1: Hierarchical Composition (Defaults Lists)

**Mechanism:** A primary config names _which option from each axis_ to compose. Each axis is a directory; each option is one small file. Framework merges them. CLI overrides win.

**Evidence:**

- **Hydra** (Meta) — [defaults list docs](https://hydra.cc/docs/advanced/defaults_list/), [experiment pattern](https://hydra.cc/docs/patterns/configuring_experiments/). Adding a variant = one file in one directory. Sweeps = `--multirun model=a,b dataset=x,y`.
- **Habitat Lab** (Meta) — embodied AI with tasks × environments × agents × sensors × datasets. Defaults list reads like an assembly manifest. Package redirection (`@habitat.simulator.agents.agent_0`) handles multiple slots. ([config README](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/README.md))
- **Fairseq** (Meta) — [Hydra integration](https://github.com/facebookresearch/fairseq/blob/main/docs/hydra_integration.md). Config groups mirror component types (model/, task/, criterion/). Registry pattern pairs each component with a dataclass.
- **NeMo** (NVIDIA) — OmegaConf + Hydra + Fiddle under the hood. ([config docs](https://docs.nvidia.com/nemotron/nightly/nemo_runspec/omegaconf.html))

**Handles explosion:** File count grows linearly with options per axis (N), not multiplicatively (N^K). New axis = new directory.

**Trade-offs:** Great for team collaboration and moderate combinatorics. Bad when axes aren't truly independent or when you need structural variation (not just value swaps).

---

### Pattern 2: Base + Overlay / Delta Patches

**Mechanism:** A complete, valid base config is the starting point. Thin overlays specify only deltas. Deep-merge applies them. Max inheritance depth ~3.

**Evidence:**

- **MMDetection** (OpenMMLab) — **872 config files** managed via 3-level `_base_` inheritance. 46 base files, 826 model-specific configs. A ResNet-101 variant is ~3 lines changing backbone depth. `_delete_=True` for structural swaps. ([config docs](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/config.html))
- **Kustomize** (Kubernetes) — bases + environment overlays. Strategic merge patches know array semantics (merge by `name` field, not replace). ([docs](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/), [tutorial](https://glasskube.dev/blog/patching-with-kustomize/))
- **Helm** — template + `values.yaml` defaults + user override files per environment. ([values files](https://helm.sh/docs/chart_template_guide/values_files/))
- **Terraform** — generic modules with input variables + environment-specific root modules + workspaces. ([module composition](https://developer.hashicorp.com/terraform/language/modules/develop/composition))

**Handles explosion:** Per-variant cost is tiny (only deltas). MMDetection's 826 configs are thin because they inherit almost everything.

**Trade-offs:** Low learning curve, diff-friendly. But list replacement (not merge) is the #1 trap — exactly the bug you hit with callbacks. Base structural changes cascade to all descendants.

---

### Pattern 3: Programmatic Config (Code-as-Config with Deferred Instantiation)

**Mechanism:** Config is written in a real language (Python, Jsonnet, CUE). Files define data structures describing what to build. A separate `instantiate()` / `build()` step creates live objects. Full language power (functions, conditionals, imports) while keeping config separate from execution.

**Evidence:**

- **Detectron2 LazyConfig** (Meta) — Python configs with `LazyCall` dicts + recursive `instantiate()`. New variant = import base + override one field in Python. Evolved from YACS because "YACS does not offer enough flexibility." ([tutorial](https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/lazyconfigs.md))
- **Fiddle** (Google/DeepMind) — `fdl.Config` wraps callable + args, `fdl.build()` instantiates recursively with memoization. Used by NeMo Run under the hood. ([repo](https://github.com/google/fiddle), [NeMo integration](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/configuration.html))
- **Jsonnet** (Databricks, Grafana) — Databricks runs **40K+ lines of Jsonnet in 1K+ files**. Parametric construction (`newShard(name, env)`) replaces copy-paste entirely. Grafana's monitoring mixins compose dashboards + alerts as Jsonnet objects. ([Databricks blog](https://medium.com/databricks-engineering/declarative-infrastructure-with-the-jsonnet-templating-language-e33d97e862fd), [Tanka](https://grafana.com/blog/2020/03/11/how-the-jsonnet-based-project-tanka-improves-kubernetes-usage/))
- **CUE** (Istio, Dagger, Mercari) — constraints and values on a single continuum. Merge is commutative + idempotent — order never matters. ([docs](https://cuelang.org/docs/concept/configuration-use-case/), [Mercari](https://engineering.mercari.com/en/blog/entry/20220127-kubernetes-configuration-management-with-cue/))

**Handles explosion:** Functions generate configs parametrically. New variation _dimensions_ can be added without restructuring. Sub-linear file growth.

**Trade-offs:** Maximum flexibility, but config files become code — harder to review, higher learning curve, debugging requires understanding the config language.

---

## How This Maps to KD-GAT

Your current system is a **hybrid of Patterns 1 and 2** — and it's hitting the known failure modes of both:

| Your system                                    | Pattern                             | Known failure mode you've hit                                                                                |
| ---------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `trainer.yaml` → stage YAML → overlay YAML     | Pattern 2 (base + overlay)          | **List replacement** — stage YAML's `callbacks:` atomically drops ModelCheckpoint (the forced-callbacks bug) |
| Stage YAMLs as independent options             | Pattern 1 (composition)             | Works, but overlays are **manually enumerated** per model×scale instead of being independent axes            |
| `pipeline.yaml` + `resources.yaml` as topology | Neither — it's parallel declaration | **Drift** — resources.yaml can fall out of sync with pipeline.yaml, only caught at dagster load time         |
| Recipe YAMLs enumerating configs               | Manual enumeration                  | Doesn't scale — each new ablation dimension multiplies the recipe entries                                    |

The key insight: **your variation axes (model_type, scale, stage, dataset) are independent**, which is the ideal case for Pattern 1. But your overlay files (`small_gat.yaml`, `large_vgae.yaml`) encode two axes (scale × model) in a single file — forcing you to enumerate the cross product manually. Separating these into independent config groups (one file per scale option, one file per model-type option) would make the system compose naturally without the file explosion.

The forced-callbacks fix (Pattern 2's list-replacement trap) is independent of any restructuring — it should go in regardless.
