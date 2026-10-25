"""LLM-as-Jev training-free recipe (arXiv:2610.02076, §3.2-3.3) on MLX.

Turns a frozen chat LLM into a decision model: given a state and K options,
return P(k | state) by scoring each candidate suffix "k]" after the assistant
prefix "Best answer: [" -- no text generation.

    python jev.py            # tiny demo
    python jev.py --check    # cached trie scoring vs. naive full recompute
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-4bit"
INSTRUCTION = "Select one option. Answer only with its bracketed numeric identifier."
ANSWER_PREFIX = "Best answer: ["


@dataclass
class Decision:
    probs: np.ndarray  # P(k | x), shape [K]
    logp: np.ndarray  # s_k = sum_t log p(c_k,t | x, c_k,<t), shape [K]
    prompt_tokens: int
    forward_calls: int  # 1 prefill + (0 or 1) batched trie step
    seconds: float
    # mlxpoc.md (Gemini) readout from the same prefill: log p of the first suffix
    # token only. With digit-level tokenizers, "1]" and "10]".."19]" share "1".
    first_logp: np.ndarray | None = None

    @property
    def best(self) -> int:
        return int(self.probs.argmax())

    @property
    def first_probs(self) -> np.ndarray:
        e = np.exp(self.first_logp - self.first_logp.max())
        return e / e.sum()


def format_user_message(state: str, options: list[str]) -> str:
    lines = [f"State: {state}", "Options:"]
    lines += [f"[{i}] {o}" for i, o in enumerate(options, 1)]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


def _tile_cache(cache, n: int):
    """Repeat a batch-1 prompt cache n times along the batch axis."""
    tiled = []
    for c in cache:
        if type(c) is KVCache:
            k, v = c.keys_and_values()
            t = KVCache()
            t.state = (mx.repeat(k, n, axis=0), mx.repeat(v, n, axis=0), c.offset)
        elif type(c) is ArraysCache:
            t = ArraysCache(len(c.cache))
            t.cache = [None if a is None else mx.repeat(a, n, axis=0) for a in c.cache]
        else:
            raise NotImplementedError(f"cache type {type(c).__name__}")
        tiled.append(t)
    return tiled


class Jev:
    def __init__(self, model_id: str = DEFAULT_MODEL, fp32: bool = True):
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        if fp32:
            # bf16 logits of magnitude ~20 have a resolution of ~0.1 nats, enough to
            # perturb suffix scores; the paper also runs inference in FP32 (App. A).
            # Quantized weights stay 4-bit; only activations/scales go FP32.
            self.model.set_dtype(mx.float32)

    # ---------- prompt / suffix construction (§3.2, App. A & C) ----------

    def prompt(self, state: str, options: list[str]) -> str:
        msgs = [{"role": "user", "content": format_user_message(state, options)}]
        chat = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        return chat + ANSWER_PREFIX

    def encode(self, text: str) -> list[int]:
        # The chat template already contains any special tokens.
        return self.tokenizer.encode(text, add_special_tokens=False)

    def suffixes(self, prompt: str, prompt_ids: list[int], k: int) -> list[list[int]]:
        """Suffix tokens of "i]" taken from tokenising prompt + suffix (App. C)."""
        out = []
        for i in range(1, k + 1):
            full = self.encode(prompt + f"{i}]")
            if full[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"prompt is not a token prefix of candidate {i}]")
            out.append(full[len(prompt_ids) :])
        return out

    # ---------- scoring ----------

    def score(self, state: str, options: list[str]) -> Decision:
        """Eq. (1): one prompt prefill + one batched forward over trie nodes."""
        t0 = time.perf_counter()
        prompt = self.prompt(state, options)
        p_ids = self.encode(prompt)
        sufs = self.suffixes(prompt, p_ids, len(options))

        cache = make_prompt_cache(self.model)
        root = self.model(mx.array([p_ids]), cache=cache)[0, -1]
        root_lp = root.astype(mx.float32) - mx.logsumexp(root.astype(mx.float32))
        calls = 1

        # Internal trie nodes = proper non-empty prefixes of each suffix. Only the
        # maximal ones need a row: earlier positions of a row give the shorter nodes.
        internal = {tuple(s[:j]) for s in sufs for j in range(1, len(s))}
        rows = [p for p in internal if not any(q != p and q[: len(p)] == p for q in internal)]
        logp = first = root_lp[mx.array([s[0] for s in sufs])]
        if rows:
            width = max(map(len, rows))
            batch = mx.array([list(r) + [0] * (width - len(r)) for r in rows])  # right-pad
            logits = self.model(batch, cache=_tile_cache(cache, len(rows))).astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1)  # [rows, width]
            calls += 1
            node = {}  # trie node path -> (row, position) whose logits condition on it
            for b, r in enumerate(rows):
                for j in range(1, len(r) + 1):
                    node.setdefault(r[:j], (b, j - 1))
            # One gather for every (candidate, step >= 1) transition.
            cand, bs, ps, toks = zip(*[
                (k, *node[tuple(s[:t])], s[t])
                for k, s in enumerate(sufs) for t in range(1, len(s))
            ])
            bs, ps, toks = mx.array(bs), mx.array(ps), mx.array(toks)
            step_lp = logits[bs, ps, toks] - lse[bs, ps]
            logp = logp + mx.zeros_like(logp).at[mx.array(cand)].add(step_lp)
        probs = mx.softmax(logp)
        mx.eval(probs, logp, first)
        return Decision(
            probs=np.array(probs), logp=np.array(logp), prompt_tokens=len(p_ids),
            forward_calls=calls, seconds=time.perf_counter() - t0, first_logp=np.array(first),
        )

    def score_naive(self, state: str, options: list[str]) -> Decision:
        """Reference: recompute prompt + full suffix for every candidate, no cache."""
        t0 = time.perf_counter()
        prompt = self.prompt(state, options)
        p_ids = self.encode(prompt)
        sufs = self.suffixes(prompt, p_ids, len(options))
        logp = []
        for s in sufs:
            ids = p_ids + s
            logits = self.model(mx.array([ids]))[0].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            n = len(p_ids)
            logp.append(sum(lp[n - 1 + t, s[t]].item() for t in range(len(s))))
        logp = np.array(logp)
        probs = np.exp(logp - logp.max())
        return Decision(
            probs=probs / probs.sum(), logp=logp, prompt_tokens=len(p_ids),
            forward_calls=len(sufs), seconds=time.perf_counter() - t0,
        )

    def generate_text(self, state: str, options: list[str], max_tokens: int = 32) -> int | None:
        """Text mode (App. A): greedy-continue after "Best answer: [" and parse "k]"."""
        from mlx_lm import generate

        out = generate(self.model, self.tokenizer, self.prompt(state, options),
                       max_tokens=max_tokens, verbose=False)
        head = out.split("]", 1)[0]
        if "]" in out and head.isdigit() and 1 <= int(head) <= len(options):
            return int(head) - 1
        return None


def _check(jev: Jev):
    cases = [
        ("User wants to change their shipping address.",
         ["order_status", "change_shipping", "cancel_order"]),
        ("My card payment was declined at the store yesterday, why?",
         [f"intent_{i}" for i in range(1, 25)] + ["declined_card_payment"]
         + [f"intent_{i}" for i in range(26, 120)]),
    ]
    for state, opts in cases:
        a, b = jev.score(state, opts), jev.score_naive(state, opts)
        print(f"K={len(opts):3d}  max|Δlogp|={np.abs(a.logp - b.logp).max():.2e}  "
              f"max|Δp|={np.abs(a.probs - b.probs).max():.2e}  "
              f"argmax {a.best + 1} vs {b.best + 1}  "
              f"cached {a.seconds * 1e3:.0f} ms ({a.forward_calls} calls) / "
              f"naive {b.seconds * 1e3:.0f} ms ({b.forward_calls} calls)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="keep bf16 activations")
    args = ap.parse_args()
    jev = Jev(args.model, fp32=not args.bf16)
    if args.check:
        _check(jev)
        return
    opts = ["order_status", "change_shipping", "cancel_order"]
    d = jev.score("User wants to change their shipping address.", opts)
    for o, p in zip(opts, d.probs):
        print(f"{p:7.2%}  {o}")


if __name__ == "__main__":
    main()
