"""Provider-agnostic LLM layer with validated structured output.

Nodes ask for a *role* ("fast" or "strong"); a provider maps the role to a concrete
model. Swapping Bedrock for Azure OpenAI is a config change, checked by the evals.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLM(Protocol):
    def complete(self, *, role: str, system: str, prompt: str) -> LLMResponse: ...


class StructuredOutputError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return text[start:end + 1]


def structured(llm: LLM, schema: type[T], *, role: str, system: str, prompt: str,
               retries: int = 1) -> tuple[T, list[LLMResponse]]:
    """Call the model and validate its output against `schema`.

    On invalid output, retry once with the validation error shown to the model.
    Raises StructuredOutputError if it still fails; callers escalate to a human.
    """
    responses: list[LLMResponse] = []
    current_prompt = prompt
    for _ in range(retries + 1):
        resp = llm.complete(role=role, system=system, prompt=current_prompt)
        responses.append(resp)
        try:
            return schema.model_validate(json.loads(_extract_json(resp.text))), responses
        except (ValueError, ValidationError) as exc:
            current_prompt = (f"{prompt}\n\nYour previous answer was invalid: {exc}. "
                              "Return only valid JSON matching the schema.")
    raise StructuredOutputError(f"{schema.__name__}: invalid output after {retries + 1} attempts")


# --------------------------------------------------------------------------- #
# Real providers. Imported lazily so the core package has no SDK dependency.
# --------------------------------------------------------------------------- #

class OpenAIChat:
    """OpenAI or Azure OpenAI chat completions. For Azure, models are deployment names."""

    def __init__(self, models: dict[str, str], *, azure: bool = False, client=None):
        if client is None:
            if azure:
                from openai import AzureOpenAI
                client = AzureOpenAI(api_version=os.environ.get("OPENAI_API_VERSION", "2024-10-21"))
            else:
                from openai import OpenAI
                client = OpenAI()
        self.client, self.models = client, models

    def complete(self, *, role: str, system: str, prompt: str) -> LLMResponse:
        model = self.models[role]
        start = time.perf_counter()
        r = self.client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return LLMResponse(r.choices[0].message.content or "", model,
                           r.usage.prompt_tokens, r.usage.completion_tokens,
                           (time.perf_counter() - start) * 1000)


class BedrockConverse:
    """Amazon Bedrock via the Converse API (works across Anthropic, Llama, Mistral, Nova models)."""

    def __init__(self, models: dict[str, str], *, client=None):
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime")
        self.client, self.models = client, models

    def complete(self, *, role: str, system: str, prompt: str) -> LLMResponse:
        model = self.models[role]
        start = time.perf_counter()
        r = self.client.converse(
            modelId=model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 1024},
        )
        text = "".join(block.get("text", "") for block in r["output"]["message"]["content"])
        return LLMResponse(text, model, r["usage"]["inputTokens"], r["usage"]["outputTokens"],
                           (time.perf_counter() - start) * 1000)


def provider_from_env(name: str) -> LLM:
    """Build a provider from environment variables. See .env.example."""
    if name == "stub":
        from .stub_llm import StubLLM
        return StubLLM()
    models = {"fast": os.environ["OPS_MODEL_FAST"], "strong": os.environ["OPS_MODEL_STRONG"]}
    if name == "openai":
        return OpenAIChat(models)
    if name == "azure":
        return OpenAIChat(models, azure=True)
    if name == "bedrock":
        return BedrockConverse(models)
    raise ValueError(f"unknown provider {name!r}")
