# 0012 - Use PBRS-compliant fusion reward

## Context

Fusion RL runs were collapsing toward benign-majority equilibria and
`alpha≈0.5` / `alpha→1.0` artifacts because the reward mixed in
terms that depended on action agreement and on the fusion weight alone.

## Decision

Use a minimal reward with:

- asymmetric classification costs for false negatives and false positives
- positive rewards for true positives and true negatives
- an attack-gated confidence bonus

Do not use reward terms that:

- reward or penalize `alpha` directly
- reward model agreement or disagreement directly
- encode an implicit prior that benign samples should dominate

## Rationale

Those shaping terms are not potential-based shaping, so they change the
objective instead of preserving it. They also make the benign-majority
equilibrium too attractive on the current datasets.

## Consequences

- Score-fusion and RL baselines should be treated as thresholding or
  calibration mechanisms, not as a place to smuggle in a better reward.
- The 18-dim fusion cache stays the supervised baseline interface.
- If a later phase needs ranking help, add it as a separate, explicit
  term instead of reintroducing agreement shaping.

## Related docs

- [`docs/reference/fusion-state.md`](../reference/fusion-state.md)
- [`docs/drafts/fusion-improvement-plan.md`](../drafts/fusion-improvement-plan.md)
