# llm-as-jev-exp — MLX proof of concept

A proof of concept (PoC) of the **training-free recipe** from
*LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When and How to Fine-Tune Them*
(Li & Wagle, [arXiv:2610.02076v3](https://arxiv.org/html/2610.02076v3)). It runs on Apple Silicon with
[`mlx-lm`](https://github.com/ml-explore/mlx-lm).

The paper has no released code. This repo rebuilds the method from §3.2–3.3 and Appendices A and C.

## The idea in one paragraph

You give a frozen chat LLM a state and K numbered options. You ask it to answer with a bracketed number,
and you **prefill the assistant turn with `Best answer: [`**. Each option k is the suffix `k]`. You score each
suffix by summing its token log-probs under teacher forcing, then softmax over all K candidates. The result
is a calibrated categorical distribution with no text generation. The suffixes form a prefix-free trie
(`1]` is never a prefix of `12]`), so this works for any K: one prompt prefill plus 1–3 short forward steps.

## Review of the Gemini sketch (`mlxpoc.md`)

The overall direction is right. MLX is a good fit, and the core is "one prefill, read the logits, softmax
over the candidates". The snippet as written is wrong in ways that matter, though:

| # | Gemini sketch | Problem | Fix in this PoC |
|---|---|---|---|
| 1 | `mlx-community/Qwen3.5-4B-Instruct-4bit` | That repo does not exist. Qwen3.5 has no `-Instruct` suffix. | `mlx-community/Qwen3.5-4B-4bit` (default), or `-8bit` / `-bf16` |
| 2 | Raw string prompt, no chat template | The paper applies the **chat template** (`add_generation_prompt=True`, `enable_thinking=False`) and places `Best answer: [` **in the assistant turn**. A raw prompt is a different (base-LM) setting. | `apply_chat_template(...)` and then append `Best answer: [` |
| 3 | `tokenizer.encode("1]")[0]`, only the first token | Qwen splits digits one per token. `1]` is `["1","]"]`, so this scores only `"1"`. **Option 1 and options 10–19 then share one logit** whenever K ≥ 10. This is the "option-10 effect" in Appendix C/H. | Full suffix scoring, Eq. (1): sum the log-probs over every token of `k]` |
| 4 | Tokenises the suffix on its own | This is wrong for SentencePiece tokenizers (they add a word-start marker), and the token boundaries can differ from the in-context tokenisation. | Take the suffix tokens from the tokenisation of `prompt + suffix`, and assert that the prompt is a token prefix (App. C) |
| 5 | Softmax over the raw single-token logits | This is correct only when every suffix is one token. | Softmax over the summed suffix log-probs `s_k` |
| 6 | Multi-digit handling is a "you could use the cache" remark | Never implemented. | Trie walk: prefill once, then **one batched forward over the internal trie nodes** with the prompt KV-cache tiled across the batch |

Verdict: the claim that a PoC is not hard holds. The sketch matches the paper only for K ≤ 9 on a base
prompt. It quietly breaks on Banking77/CLINC-style tasks, which are the paper's main selling point.

## Scope

**In scope (training-free recipe):**
- `jev.py`: the library. Chat-templated prompt, prefix-checked suffix tokenisation, trie construction,
  KV-cached parallel suffix scoring, and softmax. It returns `P(k | state)`.
- `--check`: compares the cached trie scores with a naive full recomputation of each candidate. The paper
  reports agreement within 1e-5; we expect agreement within the quantisation and numerical noise of MLX.
- `demo.py`: a few hand-written decisions (choice, yes/no "noul", ordinal score) that print distributions.
- `eval_banking77.py`: a 77-way intent eval on a sample of Banking77. It reports accuracy, ECE (15 bins)
  and Brier score, and compares the full-suffix scorer with the Gemini first-token scorer to show the
  option-10 effect.
- Optional text mode: greedy-generate after `Best answer: [` and parse `k]`, as a sanity baseline.

**Out of scope (for now):**
- The fine-tuning recipe (tree-factorised listwise loss + KL anchors + LoRA). It is feasible on MLX with a
  custom loss, but it is a separate project.
- Thinking Jev (budgeted `<think>`). It could be added later as a flag.
- Multimodal inputs and JevBench (the public items are not bundled here).

## Expected numbers (from the paper, FP32 on GPU)

| Model | Banking77 (zero-shot) |
|---|---|
| Qwen3.5-4B training-free | 69.0 % |
| Qwen3-0.6B training-free | 22.2 % |

A 4-bit MLX model on a Banking77 sample should land in the same range, not match exactly.

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx-lm datasets
python demo.py
python jev.py --check                     # cached trie vs. naive recompute
python eval_banking77.py --n 300          # sampled 77-way eval
```

## Results

_To be filled in after running the PoC._
