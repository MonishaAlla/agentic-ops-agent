# ADR-0001: Use a LangGraph state graph, not a free-form ReAct loop

**Status:** Accepted

## Context
The agent must always sanitise input, classify, check policy and get human approval
before any external action. A single ReAct loop ("think, pick a tool, repeat") leaves
the order of those steps to the model.

## Options
1. **Single ReAct agent** with all tools. Least code; least control. The model can skip
   the policy check or call `send_reply` early, and loops are possible.
2. **Multi-agent framework (CrewAI / AutoGen style)** with role agents. Good for open-ended
   collaboration; adds inter-agent chatter, cost and non-determinism this task doesn't need.
3. **LangGraph `StateGraph`** with fixed stages and model freedom inside each stage.
4. **Plain Python pipeline** with no framework.

## Decision
Option 3. The workflow has a known shape, so the graph encodes it. LangGraph also gives
us, without custom code: typed shared state, conditional edges, `interrupt` for
human-in-the-loop, and checkpointing so a run paused for approval can resume hours later.

## Consequences
- The stage order is guaranteed by code, which is what auditors and risk teams want.
- Less flexible for truly open-ended requests; those go to the human queue by design.
- We depend on LangGraph's API. Mitigation: nodes are plain functions over the state,
  so moving to option 4 later is a re-wiring job, not a rewrite.
