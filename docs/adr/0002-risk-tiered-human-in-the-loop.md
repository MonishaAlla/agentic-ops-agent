# ADR-0002: Tier tools by risk and enforce the tiers in code

**Status:** Accepted

## Context
Asking a human to approve every step makes the agent pointless; approving nothing is
unacceptable for customer-facing actions in a regulated business. Prompt rules alone
("never send without approval") can be overridden by injected text in an email.

## Decision
Every tool declares a risk tier: `READ`, `INTERNAL_WRITE` or `EXTERNAL`.
- `READ` tools run freely.
- `INTERNAL_WRITE` tools run and are written to the audit log.
- `EXTERNAL` tools can only run after the graph has been interrupted and a human has
  approved the exact payload. The tool executor refuses an `EXTERNAL` call without an
  approval record that matches the payload hash.

## Consequences
- A successful prompt injection can at worst cause a bad draft, which a human sees.
- Approval is bound to the payload, so a draft cannot change after approval.
- Human effort scales with external actions only, not with every step.
- Moving an intent to auto-send later is an explicit, reviewable config change.
