"""Shared Claude wrapper: async calls, concurrency cap, cost tracking, hard budget cap, mock mode.

Usage:
    llm = LLM(model="claude-haiku-5-5", budget_usd=25, max_concurrency=16, mock=False)
    text = await llm.ask(system, user, mock_fn=lambda: '{"step1": "soak", "step2": "dry"}')
    data = await llm.ask_json(system, user, mock_fn=lambda: {"step1": "soak", "step2": "dry"})

In mock mode no network call is made and `mock_fn()` supplies the reply, so every
script can be tested end to end at zero cost. `ask_json` returns {} when the reply
cannot be parsed; callers must fall back to a default action and count it.
"""
import asyncio
import json
import re

import anthropic

# USD per 1M tokens (input, output). Source: claude-api skill model table, 2026-11-03.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class BudgetExceeded(RuntimeError):
    """Raised before a call once tracked spend has reached the cap."""


class LLM:
    def __init__(self, model="claude-haiku-5-5", budget_usd=25.0, max_concurrency=16,
                 mock=False, effort="low", thinking=None):
        if model not in PRICES:
            raise ValueError(f"No price known for {model}; add it to PRICES.")
        self.model = model
        self.budget_usd = float(budget_usd)
        self.mock = mock
        self.effort = effort
        self.thinking = thinking  # None = model default; {"type": "disabled"} turns hidden reasoning off
        self.sem = asyncio.Semaphore(max_concurrency)
        # gzip only: this machine's zstandard build breaks httpx2's default decompression.
        self.client = None if mock else anthropic.AsyncAnthropic(
            max_retries=8, default_headers={"Accept-Encoding": "gzip"})
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.failures = 0     # calls that returned nothing usable after retries
        self.parse_fail = 0   # ask_json replies that could not be parsed

    def usage(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 4),
                "failures": self.failures, "parse_fail": self.parse_fail}

    async def ask(self, system: str, user: str, mock_fn=None, max_tokens: int = 2000) -> str:
        if self.mock:
            self.calls += 1
            out = mock_fn() if mock_fn else ""
            return out if isinstance(out, str) else json.dumps(out)
        if self.cost_usd >= self.budget_usd:
            raise BudgetExceeded(f"Spend ${self.cost_usd:.2f} reached cap ${self.budget_usd:.2f}")
        async with self.sem:
            # Re-check after waiting: calls queued behind the semaphore must not overshoot the cap.
            if self.cost_usd >= self.budget_usd:
                raise BudgetExceeded(f"Spend ${self.cost_usd:.2f} reached cap ${self.budget_usd:.2f}")
            for attempt in range(4):
                try:
                    resp = await self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        output_config={"effort": self.effort},
                        **({"thinking": self.thinking} if self.thinking else {}),
                    )
                except anthropic.BadRequestError:
                    raise  # a malformed request will not fix itself
                except anthropic.AuthenticationError:
                    raise
                except anthropic.RateLimitError:
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                except anthropic.APIStatusError as e:
                    if e.status_code >= 500:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    raise
                except anthropic.APIConnectionError:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                self.calls += 1
                p_in, p_out = PRICES[self.model]
                self.input_tokens += resp.usage.input_tokens
                self.output_tokens += resp.usage.output_tokens
                self.cost_usd += (resp.usage.input_tokens * p_in
                                  + resp.usage.output_tokens * p_out) / 1e6
                if resp.stop_reason == "refusal":
                    self.failures += 1
                    return ""
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            self.failures += 1
            return ""

    async def ask_json(self, system: str, user: str, mock_fn=None, max_tokens: int = 2000) -> dict:
        """Ask for a JSON object. Returns {} if no object can be parsed after one retry."""
        for _ in range(2):
            text = await self.ask(system, user, mock_fn=mock_fn, max_tokens=max_tokens)
            data = parse_json_object(text)
            if data is not None:
                return data
        self.parse_fail += 1
        return {}


def parse_json_object(text: str):
    """First JSON object found in the text, or None."""
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None
