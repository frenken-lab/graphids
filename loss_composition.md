# Loss Composition in Research Codebases

## Motivation

Complex models stacking multiple learning paradigms (e.g. VGAE + GAT + PINN + federated regularization) 
produce loss functions of the form:

```
L_total = λ_task · L_task + λ_kl · L_kl + λ_pinn · L_pinn + λ_fed · L_fed + ...
```

A naive weighted sum in the training loop breaks down as complexity grows:

- **Gradient conflicts** — PINN physics residuals and KL terms routinely push parameters in opposing directions
- **Scale mismatch** — terms can differ by orders of magnitude, drowning out weaker signals
- **Ablation friction** — toggling a term requires editing training loop logic
- **Silent failures** — individual terms can go to zero or diverge invisibly if only the total is logged

---

## Directory Structure

```
losses/
├── __init__.py           # exports CompositeLoss + all terms
├── base.py               # LossTerm base class
├── task.py               # CrossEntropy / node classification
├── kl.py                 # VGAE KL divergence
├── pinn.py               # Physics residual
├── federated.py          # FedProx regularization
└── composer.py           # CompositeLoss + optional gradient surgery
```

---

## Base Class — `losses/base.py`

Every loss term is an `nn.Module` with a scalar weight. It reads whatever it needs 
directly from a shared `TensorDict` (or plain dict), keeping terms fully decoupled.

```python
from abc import abstractmethod
import torch.nn as nn

class LossTerm(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    @abstractmethod
    def forward(self, td) -> torch.Tensor:
        """Returns unreduced scalar loss."""
        ...
```

---

## Example Terms

**`losses/kl.py`**
```python
class KLLoss(LossTerm):
    def forward(self, td):
        mu, logstd = td["mu"], td["logstd"]
        return -0.5 * (1 + logstd - mu**2 - logstd.exp()).mean()
```

**`losses/pinn.py`**
```python
class PINNLoss(LossTerm):
    def forward(self, td):
        return physics_residual(td["pinn_out"], td["coords"])
```

**`losses/federated.py`**
```python
class FedProxLoss(LossTerm):
    """Penalizes deviation from global model. Call after aggregation rounds only."""
    def __init__(self, global_model, mu=0.01, weight=1.0):
        super().__init__(weight)
        self.global_model = global_model
        self.mu = mu

    def forward(self, td):
        local_params = td["local_params"]   # passed in or read from model directly
        return (self.mu / 2) * sum(
            (p - p_g).norm() ** 2
            for p, p_g in zip(local_params, self.global_model.parameters())
        )
```

---

## Composer — `losses/composer.py`

```python
class CompositeLoss(nn.Module):
    def __init__(self, terms: dict[str, LossTerm]):
        super().__init__()
        self.terms = nn.ModuleDict(terms)

    def forward(self, td):
        losses = {name: term(td) for name, term in self.terms.items()}
        total = sum(term.weight * losses[name]
                    for name, term in self.terms.items())
        return total, losses   # always return individual terms for logging
```

---

## Configuration

Weights live in config, not in code. Using Hydra:

**`configs/losses.yaml`**
```yaml
losses:
  task:
    weight: 1.0
  kl:
    weight: 1.0e-3
  pinn:
    weight: 0.1
  federated:
    weight: 0.5
    mu: 0.01
```

Instantiation at training time:

```python
loss_fn = CompositeLoss({
    "task":      TaskLoss(weight=cfg.losses.task.weight),
    "kl":        KLLoss(weight=cfg.losses.kl.weight),
    "pinn":      PINNLoss(weight=cfg.losses.pinn.weight),
    "federated": FedProxLoss(global_model, weight=cfg.losses.federated.weight),
})
```

To ablate a term: set its weight to `0.0` in config — no code changes.

---

## Training Loop Integration

```python
total, loss_dict = loss_fn(td)
total.backward()
optimizer.step()

# Log everything — non-negotiable
wandb.log({f"loss/{k}": v.item() for k, v in loss_dict.items()})
wandb.log({"loss/total": total.item()})
```

---

## Gradient Conflict Handling

For known conflicting paradigms (PINN + KL is a common offender), check cosine 
similarity between gradients early in training before committing to fixed weights:

```python
# Diagnostic — run for first ~100 steps
g_task = get_grads(losses["task"], model)
g_pinn = get_grads(losses["pinn"], model)
cos_sim = F.cosine_similarity(g_task, g_pinn, dim=0).mean()
# cos_sim << 0 → conflicting, consider PCGrad or GradNorm
```

If conflicts are severe, drop in **PCGrad** (gradient surgery) or **GradNorm** as a 
wrapper around `CompositeLoss` — the modular structure makes this a one-line swap 
in the training loop without touching individual loss definitions.

---

## Key Rules

1. **Every term is a module** — weights, buffers, and sub-networks stay encapsulated
2. **Always log individually** — never only log `loss/total`
3. **Weights live in config** — ablation is a config change, not a code change
4. **Federated terms have different cadence** — only apply after aggregation rounds, 
   guard with a step counter or a flag
5. **Check gradients early** — cosine similarity diagnostic in the first epoch saves 
   weeks of debugging later
