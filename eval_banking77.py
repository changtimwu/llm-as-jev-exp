"""Zero-shot 77-way Banking77 intent eval for the LLM-as-Jev readout (paper §4.2).

Compares, from the same prefill:
  * jev   -- full candidate-suffix scoring, Eq. (1)        (the paper)
  * first -- first-suffix-token softmax                    (mlxpoc.md sketch)
and optionally text mode (greedy generation after "Best answer: [").

    python eval_banking77.py --n 300
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
from datasets import load_dataset

from jev import DEFAULT_MODEL, Jev


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def metrics(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(1)
    correct = (pred == y).astype(float)
    onehot = np.eye(P.shape[1])[y]
    return {
        "acc": float(correct.mean()),
        "ece": ece(P.max(1), correct),
        "brier": float(((P - onehot) ** 2).sum(1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=300, help="test items to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", action="store_true", help="also run text (generation) mode")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = load_dataset("mteb/banking77", split="test")
    names = dict(zip(ds["label"], ds["label_text"]))
    options = [names[i] for i in range(len(names))]  # canonical label order
    idx = list(range(len(ds)))
    if args.n:
        idx = random.Random(args.seed).sample(idx, args.n)

    jev = Jev(args.model, fp32=not args.bf16)
    P_jev, P_first, y, secs, text_pred = [], [], [], [], []
    t0 = time.time()
    for n, i in enumerate(idx, 1):
        ex = ds[i]
        d = jev.score(ex["text"], options)
        P_jev.append(d.probs)
        P_first.append(d.first_probs)
        y.append(ex["label"])
        secs.append(d.seconds)
        if args.text:
            text_pred.append(jev.generate_text(ex["text"], options))
        if n % 25 == 0 or n == len(idx):
            acc = np.mean(np.array(P_jev).argmax(1) == np.array(y))
            print(f"[{n}/{len(idx)}] jev acc so far {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    P_jev, P_first, y = np.array(P_jev), np.array(P_first), np.array(y)
    res = {
        "model": args.model,
        "precision": "bf16" if args.bf16 else "fp32-activations",
        "n": len(idx),
        "prompt_tokens_median": int(np.median([len(jev.encode(jev.prompt("x", options)))])),
        "latency_ms_median": float(np.median(secs) * 1e3),
        "jev": metrics(P_jev, y),
        "first_token": metrics(P_first, y),
        # distinct first-token values available to the first-token readout
        "first_token_distinct_scores": int(len({round(float(v), 6) for v in P_first[0]})),
    }
    if args.text:
        tp = np.array([-1 if p is None else p for p in text_pred])
        res["text"] = {"acc": float((tp == y).mean()), "parse_fail": int((tp == -1).sum()),
                       "agree_with_jev": float((tp == P_jev.argmax(1)).mean())}
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
