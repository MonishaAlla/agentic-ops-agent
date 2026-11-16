# ADR-0004: Route each step to the cheapest model that passes its evals

**Status:** Accepted

## Context
Classification is a short structured-output task. Drafting a customer reply needs better
writing and reasoning. Using the largest model for both wastes money on the easy step;
using the smallest for both hurts reply quality.

## Decision
- Each node names a model *role* (`fast`, `strong`), not a model ID.
- Config maps roles to concrete models per provider.
- A role can only be moved to a cheaper model if that node's eval metrics stay within
  tolerance on the eval set.

## Consequences
- Cost per message drops for the majority of traffic that only needs `fast`.
- Two models to monitor instead of one.
- Provider portability: switching provider is a config change, verified by the eval run.
