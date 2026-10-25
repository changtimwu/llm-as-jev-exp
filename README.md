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

Hardware: Apple M1 Max (32 GB), mlx 0.32.4, mlx-lm 0.32.2. Raw outputs are in [`results/`](results/).

### 1. Cached trie scoring is exact (`python jev.py --check`)

| K | max \|Δ log p\| vs naive | argmax | cached | naive |
|---|---|---|---|---|
| 3 | 0 | same | 191 ms (2 forward calls) | 326 ms (3 calls) |
| 119 | 3.8e-5 | same | 2.5 s (2 calls) | 348 s (119 calls) |

One prefill plus one batched trie step reproduces the full per-candidate recomputation within 4e-5 nats,
about **140× faster** at K=119. The paper reports agreement within 1e-5.

**Precision gotcha.** With the model's native bf16 activations, the same check drifts by up to **0.15 nats**.
bf16 logits of magnitude about 20 have a resolution of roughly 0.1, and a batched forward pass rounds
differently from a single-row one. Casting activations to FP32 (`model.set_dtype(mx.float32)`, with the
weights still 4-bit) fixes it at no measurable cost. FP32 is the default; `--bf16` turns it off. The paper
also runs inference in FP32.

### 2. Demo: the option-10 effect (`python demo.py`)

For K ≤ 9 the first-token readout from the Gemini sketch is almost identical to full scoring. At K=12 it
breaks. Here the correct answer is option 10:

```
[choice, K=12 (option 10 is correct)] Which month has the fewest days?
     [ 1]   1.09%   (first-token  24.81%)  January
  <- [10]  95.81%   (first-token  24.81%)  February
     [11]   0.03%   (first-token  24.81%)  November
     [12]   0.14%   (first-token  24.81%)  December
```

The first-token readout cannot tell `1]`, `10]`, `11]` and `12]` apart, because all of them start with
the token `1`. Full output: [`results/demo_output.txt`](results/demo_output.txt).

### 3. Banking77, 77-way zero-shot (n = 300, seed 0, `mteb/banking77` test)

| Model | Readout | Acc | ECE (15 bins) | Brier | Paper (full test) |
|---|---|---|---|---|---|
| Qwen3.5-4B-4bit, FP32 act. | **full suffix (paper)** | **67.3 %** | 0.076 | 0.442 | 69.0 % |
| | text mode (greedy) | 67.7 % (0 parse fails, 98 % agree) | – | – | 68.5 % |
| | first token (Gemini sketch) | 11.0 % | 0.034 | 0.933 | – |
| Qwen3.5-4B-4bit, bf16 act. | full suffix | 67.3 % | 0.080 | 0.442 | – |
| | first token | 11.0 % | 0.035 | 0.933 | – |
| Qwen3-0.6B-4bit, FP32 act. | **full suffix (paper)** | **21.3 %** | 0.363 | 1.089 | 22.2 % |
| | text mode (greedy) | 21.7 % (0 parse fails, 94 % agree) | – | – | 21.9 % |
| | first token | 8.7 % | 0.027 | 1.004 | – |

Median scoring latency for a prompt of about 710 tokens: **2.3 s** for 4B FP32, 2.0 s for 4B bf16, and
0.46 s for 0.6B. The paper reports 194 ms (4B) and 76 ms (0.6B) in FP32 on an RTX A5500. Nothing here is
optimised for speed.

**Takeaways**
- A 4-bit MLX port reproduces the paper's training-free Banking77 numbers within sampling noise. With
  n=300 the 95 % CI is about ±5 pts, which covers 69.0 % and 22.2 %.
- The paper's suffix readout and greedy text mode agree on 94–98 % of items, as the paper reports
  (generation ≈ readout). The readout also returns a full distribution for free.
- On high-cardinality tasks the first-token readout is unusable: it only has 9 distinct scores for 77
  options. Its low ECE is not a sign of good calibration. Its probabilities are close to uniform, so
  confidence is low everywhere.
- Calibration at 4B is reasonable out of the box (ECE 0.076). At 0.6B it is poor (0.363), which matches
  the paper's finding that small models need fine-tuning.
- bf16 vs FP32 did not change Banking77 accuracy at all. The precision gotcha matters for exact scores
  and probabilities, not for argmax on this task.

### Caveats
- 4-bit quantized weights, not the paper's FP32 weights. A 300-item sample, not the full 3,080 items.
  Raw label names (`card_arrival`) are used as option text in canonical label order.
- JevBench, the fine-tuning recipe, Thinking Jev and image inputs are not reproduced.
