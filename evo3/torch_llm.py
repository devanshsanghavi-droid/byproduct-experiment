"""PyTorch backend with one LoRA adapter (used on Modal, CPU or GPU). Same interface as LocalLLM.

On a GPU the weights stay float32 and the forward pass runs in bfloat16 autocast."""
import random
import time

from evo3.local import LocalLLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOKEN_BUDGET = 3200   # prompt tokens per sub-batch; keeps the first-pass scores under about 2 GB


class TorchLLM(LocalLLM):
    def __init__(self, model=MODEL, rank=8, alpha=16, max_tokens=80, batch=32, seed=0, threads=None, device="cpu"):
        LocalLLM.__init__(self, model=model, rank=rank, max_tokens=max_tokens, batch=batch, mock=True)
        self.mock = False
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        try:
            base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
        self.model = get_peft_model(base, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        self.device, self.gpu = device, device != "cpu"
        self.model.to(device)
        self.model.eval()
        self.token_budget = 12000 if self.gpu else TOKEN_BUDGET
        if self.gpu:
            self.batch = 64
        self.config = {"model": model, "rank": rank, "alpha": alpha, "max_tokens": max_tokens,
                       "backend": "torch-" + device, "threads": torch.get_num_threads(),
                       "gpu": torch.cuda.get_device_name(0) if self.gpu else None}

    def _chat(self, system, user) -> str:
        return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            add_generation_prompt=True, tokenize=False)

    def _flush(self):
        torch = self.torch
        batch, self.pending = self.pending, []
        texts = [self._chat(s, u) for s, u, _, _ in batch]
        lens = [len(self.tok(x, add_special_tokens=False)["input_ids"]) for x in texts]
        order = sorted(range(len(batch)), key=lambda i: lens[i])
        out, t = [None] * len(batch), time.time()
        kw = (dict(do_sample=True, temperature=self.temp, top_p=1.0, top_k=0) if self.temp > 0
              else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
        # generate() scores the whole vocabulary at every prompt position on its first pass
        # (about 0.6 MB per token), so each sub-batch is capped at TOKEN_BUDGET prompt tokens.
        groups, cur = [], []
        for i in order:
            if cur and (len(cur) >= self.batch or (len(cur) + 1) * lens[i] > self.token_budget):
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        for idx in groups:
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.gpu):
                gen = self.model.generate(**enc, max_new_tokens=max(batch[i][2] for i in idx),
                                          repetition_penalty=1.0, pad_token_id=self.tok.pad_token_id, **kw)
            new = gen[:, enc["input_ids"].shape[1]:].cpu()
            for j, i in enumerate(idx):
                out[i] = self.tok.decode(new[j], skip_special_tokens=True)
            self.prompt_tokens += int(enc["attention_mask"].sum())
            self.gen_tokens += int((new != self.tok.pad_token_id).sum())
        self.gen_seconds += time.time() - t
        for (_, _, _, fut), text in zip(batch, out):
            fut.set_result(text)

    def train(self, examples, iters, lr=1e-4, batch_size=8, adapter_file=None, seed=0) -> list:
        if not examples or iters <= 0:
            return []
        torch, rng, pad = self.torch, random.Random(seed), self.tok.pad_token_id
        items = []
        for e in examples:
            p = self.tok(self._chat(e["system"], e["user"]), add_special_tokens=False)["input_ids"]
            o = self.tok(e["out"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
            items.append((p, o))
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        self.model.train()
        causal = self.model.get_base_model()   # LoRA layers are injected in place, so this path still trains them
        t, losses, acc, k, pool = time.time(), [], 0.0, max(1, iters // 5), []
        for it in range(1, iters + 1):
            while len(pool) < batch_size:
                pool += rng.sample(items, len(items))
            chunk, pool = pool[:batch_size], pool[batch_size:]
            L = max(len(p) + len(o) for p, o in chunk)
            ids = torch.full((batch_size, L), pad)
            lab = torch.full((batch_size, L), -100)
            att = torch.zeros((batch_size, L), dtype=torch.long)
            for j, (p, o) in enumerate(chunk):
                n = len(p) + len(o)
                ids[j, :n] = torch.tensor(p + o)
                lab[j, len(p):n] = torch.tensor(o)
                att[j, :n] = 1
            # Vocabulary scores only at reply positions: scoring every position needs gigabytes per batch.
            ids, lab, att = ids.to(self.device), lab.to(self.device), att.to(self.device)
            micro = getattr(self, "micro", None)
            if not micro or micro >= batch_size:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.gpu):
                    h = causal.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
                    sel = lab[:, 1:] != -100
                    logits = causal.lm_head(h[sel])
                loss = torch.nn.functional.cross_entropy(logits.float(), lab[:, 1:][sel])
                loss.backward()
                step_loss = float(loss)
            else:
                # Same gradient in less memory: the batch goes through in groups of `micro`; each group's summed
                # reply-token loss is divided by the token count of the WHOLE batch, and the gradients add up.
                ntok = int((lab[:, 1:] != -100).sum())
                while True:
                    step_loss, oom = 0.0, False
                    try:
                        for a in range(0, batch_size, micro):
                            n = int(att[a:a + micro].sum(1).max())     # padding is on the right, so trimming changes nothing
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.gpu):
                                h = causal.model(input_ids=ids[a:a + micro, :n], attention_mask=att[a:a + micro, :n]).last_hidden_state[:, :-1]
                                sel = lab[a:a + micro, 1:n] != -100
                                logits = causal.lm_head(h[sel])
                            part = torch.nn.functional.cross_entropy(logits.float(), lab[a:a + micro, 1:n][sel], reduction="sum") / ntok
                            part.backward()
                            step_loss += float(part)
                            del h, logits, part
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                    if not oom:
                        break
                    if micro == 1:
                        raise RuntimeError("out of GPU memory with one example per pass")
                    opt.zero_grad(set_to_none=True)
                    h = logits = part = None
                    torch.cuda.empty_cache()
                    micro = max(1, micro // 2)
                    self.micro_splits = getattr(self, "micro_splits", 0) + 1
            opt.step()
            opt.zero_grad()
            acc += step_loss
            if it % k == 0 or it == iters:
                losses.append(round(acc / (k if it % k == 0 else it % k), 4))
                acc = 0.0
        self.train_seconds += time.time() - t
        self.model.eval()
        if adapter_file:
            self.save_adapter(adapter_file)
        return losses

    def adapter_state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items() if "lora_" in k}

    def save_adapter(self, path):
        self.torch.save(self.adapter_state(), path)

    def load_adapter(self, path):
        self.model.load_state_dict(self.torch.load(path, map_location=self.device), strict=False)
        self.model.eval()
