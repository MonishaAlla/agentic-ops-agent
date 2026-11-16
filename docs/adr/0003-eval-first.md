# ADR-0003: Build the eval suite before tuning prompts, and gate CI on it

**Status:** Accepted

## Context
Prompt and model changes have non-local effects: a tweak that fixes one intent can break
another. Without a fixed eval set, every change is judged by a few hand-picked examples.

## Decision
- Keep a labelled, versioned eval dataset in the repo (synthetic, no customer data).
- Measure quality (intent F1, groundedness), safety (escalation recall, injection block
  rate) and efficiency (cost, latency) on every run.
- Safety metrics are **gates**: CI fails if they fall below threshold.
  Quality and efficiency metrics are **tracked** and reported in the PR.
- LLM-as-judge scores are calibrated against a hand-labelled subset; if agreement drops
  below 85%, the judge prompt is fixed before its scores are trusted.

## Consequences
- Every prompt or model change comes with numbers.
- Model swaps (e.g. Bedrock to Azure OpenAI) become a measured decision, not a guess.
- Eval runs cost money; the CI suite uses a fixed sample and the full set runs nightly.
