"""Local open-weights model with one LoRA adapter (mlx-lm), version 3.

Same `ask_json(system, user, mock_fn)` interface as evo.llm.LLM, so the version 2 world
and test code runs unchanged. Calls made by many coroutines are collected and generated
in one batch:

    llm = LocalLLM()
    results = asyncio.run(llm.run([life(llm.proxy(log), ...) for ...]))

`llm.train(examples, iters)` continues the same adapter on (system, user, out) examples.
The loss covers only the reply tokens.
"""
import asyncio
import json
import os
import re
import time

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


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


class Proxy:
    """An llm handle that records every exchange into `log` (one agent's transcript)."""

    def __init__(self, llm, log):
        self.llm, self.log = llm, log

    async def ask_json(self, system, user, mock_fn=None, max_tokens=None):
        return await self.llm.ask_json(system, user, mock_fn, max_tokens, log=self.log)


class LocalLLM:
    def __init__(self, model=MODEL, rank=8, scale=20.0, layers=16, max_tokens=80,
                 batch=64, seed=0, mock=False):
        self.mock, self.max_tokens, self.batch = mock, max_tokens, batch
        self.temp = 1.0
        self.pending = []
        self.calls = self.parse_fail = 0
        self.prompt_tokens = self.gen_tokens = 0
        self.gen_seconds = self.train_seconds = 0.0
        self.config = {"model": model, "rank": rank, "scale": scale, "layers": layers,
                       "max_tokens": max_tokens}
        if mock:
            return
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        self.mx = mx
        self.model, self.tok = load(model)
        self.model.freeze()
        linear_to_lora_layers(self.model, layers, {"rank": rank, "scale": scale, "dropout": 0.0})
        self.model.eval()
        mx.random.seed(seed)

    # ---- generation -------------------------------------------------------------
    def proxy(self, log):
        return Proxy(self, log)

    async def ask_json(self, system, user, mock_fn=None, max_tokens=None, log=None) -> dict:
        if self.mock:
            out = mock_fn() if mock_fn else {}
            text = out if isinstance(out, str) else json.dumps(out)
        else:
            fut = asyncio.get_running_loop().create_future()
            self.pending.append((system, user, max_tokens or self.max_tokens, fut))
            text = await fut
        self.calls += 1
        d = parse_json_object(text)
        if not isinstance(d, dict):
            self.parse_fail += 1
            d = {"_raw": text}   # callers that accept a bare answer can read it here
        if log is not None:
            log.append({"system": system, "user": user, "out": text})
        return d

    async def run(self, coros) -> list:
        """Run coroutines together; generate whenever all of them are waiting on the model."""
        tasks = [asyncio.ensure_future(c) for c in coros]
        last, still = -1, 0
        while not all(t.done() for t in tasks):
            await asyncio.sleep(0)
            n = len(self.pending)
            if n and n == last:
                still += 1
                if still >= 3:
                    self._flush()
                    last, still = -1, 0
            else:
                last, still = n, 0
        return [t.result() for t in tasks]

    def _flush(self):
        from mlx_lm.generate import batch_generate
        from mlx_lm.sample_utils import make_sampler
        batch, self.pending = self.pending, []
        prompts = [self.tok.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            add_generation_prompt=True) for s, u, _, _ in batch]
        kw = {"sampler": make_sampler(temp=self.temp)} if self.temp > 0 else {}
        t = time.time()
        r = batch_generate(self.model, self.tok, prompts, max_tokens=[m for _, _, m, _ in batch],
                           completion_batch_size=self.batch, prefill_batch_size=self.batch, **kw)
        self.gen_seconds += time.time() - t
        self.prompt_tokens += r.stats.prompt_tokens
        self.gen_tokens += r.stats.generation_tokens
        for (_, _, _, fut), text in zip(batch, r.texts):
            fut.set_result(text)

    # ---- training ---------------------------------------------------------------
    def train(self, examples, iters, lr=1e-4, batch_size=8, adapter_file=None, seed=0) -> list:
        """Continue the adapter on examples [{system, user, out}]. Returns reported losses."""
        if self.mock or not examples or iters <= 0:
            return []
        import numpy as np
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
        from mlx_lm.tuner.trainer import default_loss, iterate_batches
        mx = self.mx
        np.random.seed(seed)
        data = [{"messages": [{"role": "system", "content": e["system"]},
                              {"role": "user", "content": e["user"]},
                              {"role": "assistant", "content": e["out"]}]} for e in examples]
        while len(data) < batch_size:
            data = data + data
        ds = CacheDataset(ChatDataset(data, self.tok, mask_prompt=True))
        opt = optim.Adam(learning_rate=lr)
        grad_fn = nn.value_and_grad(self.model, default_loss)
        self.model.train()
        t, losses, acc, k = time.time(), [], 0.0, max(1, iters // 5)
        # mlx-lm's own train() is not used: it starts MPI, which aborts with this machine's MPICH.
        for it, batch in zip(range(1, iters + 1),
                             iterate_batches(ds, batch_size, 1024, loop=True, comm_group=None)):
            (loss, _), grad = grad_fn(self.model, *batch)
            opt.update(self.model, grad)
            mx.eval(self.model.parameters(), opt.state, loss)
            acc += loss.item()
            if it % k == 0 or it == iters:
                n = k if it % k == 0 else it % k
                losses.append(round(acc / n, 4))
                acc = 0.0
        self.train_seconds += time.time() - t
        self.model.eval()
        if adapter_file:
            os.makedirs(os.path.dirname(adapter_file) or ".", exist_ok=True)
            mx.save_safetensors(adapter_file, dict(tree_flatten(self.model.trainable_parameters())))
        return losses

    def load_adapter(self, path):
        self.model.load_weights(path, strict=False)
        self.model.eval()

    def usage(self) -> dict:
        return {"calls": self.calls, "parse_fail": self.parse_fail,
                "prompt_tokens": self.prompt_tokens, "gen_tokens": self.gen_tokens,
                "gen_seconds": round(self.gen_seconds, 1), "train_seconds": round(self.train_seconds, 1),
                "calls_per_second": round(self.calls / self.gen_seconds, 2) if self.gen_seconds else None}
