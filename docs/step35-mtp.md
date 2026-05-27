# Step 3.5 Flash MTP

This branch is based on StepFun's Step 3.5 MTP llama.cpp fork:

- upstream fork: https://github.com/stepfun-ai/llama.cpp
- upstream branch: `step3p5-mtp`
- base commit: `a2f5ec441` (`recover requires_dft lost in rebase`)

The StepFun branch added Step 3.5 Flash runtime/model-loading support, MTP speculative support, SWA KV rollback, prompt-cache plumbing, and MTP quantization/conversion support. This branch layers additional server/runtime handling on top: prompt-cache restore behavior for MTP, SWA prompt-cache threshold fixes, MTP prompt-cache tail hidden-row handling, optional server-side p/q acceptance, a Step thinking-template flag, and benchmark notes.

The implementation is experimental same-GGUF MTP support for Step 3.5 Flash models that contain `step35.nextn_predict_layers`.

Tested server path:

```bash
./build/bin/llama-server \
  -m /path/to/Step-3.5-Flash-MTP-IQ4_XS-3.90BPW-Q8_MTP.gguf \
  -mtp \
  --draft 1 \
  -np 1 \
  -ngl 99
```

## Current Recommendation

Use `-mtp --draft 1` for the published Step 3.5 Flash MTP GGUFs associated with this fork.

On an Apple M3 Max test setup, short server checks consistently showed `--draft 1` as the best default for the published one-nextn GGUFs. Deeper recurrent drafts accepted more total draft tokens in some runs, but their extra MTP and verification work did not pay for itself.

A later controlled 384-token check on the `IQ3_S-3.64BPW-Q8_MTP` quant showed:

| Runtime | Prompt cache path | Decode speed | Draft acceptance |
| --- | --- | ---: | ---: |
| This fork, no MTP | default cache path | 26.87 t/s | - |
| This fork, `-mtp --draft 1` | default cache path | 32.55 t/s | 168/214, 78.5% |
| StepFun `step3p5-mtp`, `-mtp --draft 1` | default cache path | 27.10 t/s | disabled by prompt-cache path |
| StepFun `step3p5-mtp`, `-mtp --draft 1 --cache-ram 0` | prompt cache disabled | 33.51 t/s | 168/214, 78.5% |

That comparison is the practical reason for the prompt-cache/MTP handling in this branch:
the StepFun branch can run MTP, but its server prompt-cache path disables MTP in
this scenario. This fork resets the MTP draft-side state after prompt-cache
restore and resumes speculation once fresh target hidden state is available.

The tested GGUF reports:

```text
step35.nextn_predict_layers = 1
```

## Published GGUFs

The published GGUFs use these public names:

| File | Notes |
| --- | --- |
| `Step-3.5-Flash-MTP-IQ4_XS-3.90BPW-Q8_MTP.gguf` | AesSedai-style mixed expert layout; MTP/nextn tensors kept Q8. |
| `Step-3.5-Flash-MTP-IQ3_S-3.64BPW-Q8_MTP.gguf` | Smaller custom IQ3_S expert layout; MTP/nextn tensors kept Q8. |
| `Step-3.5-Flash-MTP-IQ3_XXS-3.27BPW-Q8_MTP.gguf` | Smallest custom IQ3_XXS expert layout; MTP/nextn tensors kept Q8. |

The imatrix used for these quantizations came from Bartowski's Step 3.5 Flash GGUF work, not from this fork. Credit it separately when publishing model cards.

`--draft 2` and deeper drafts reuse the single MTP layer recurrently. They are not true multi-head MTP for this model file. The runtime maps draft step `k` to nextn layer `base + k` when a future Step GGUF exposes multiple nextn layers, while clamping to the last available layer if the requested draft depth is larger than the model supports.

In a fresh four-prompt matrix on the `IQ3_S-3.64BPW-Q8_MTP` quant, with `temp 0.6`, `n_predict=128`, `--cache-ram 0`, and request-level dotted keys `"speculative.n_max"` and `"speculative.pq_accept"`, the averages were:

| `n_max` | p/q accept | avg tok/s | avg acceptance |
| --- | --- | ---: | ---: |
| 1 | off | 30.38 | 0.690 |
| 1 | on | 28.10 | 0.696 |
| 2 | off | 25.68 | 0.618 |
| 2 | on | 24.93 | 0.578 |
| 3 | off | 24.06 | 0.585 |
| 3 | on | 24.04 | 0.564 |
| 4 | off | 23.36 | 0.576 |
| 4 | on | 22.04 | 0.562 |

This small matrix is not a benchmark suite, but it is enough to justify `--draft 1` as the default recommendation for these one-nextn GGUFs. `--draft 2` and deeper remain useful diagnostics for future GGUFs with more trained nextn layers, but they were slower in this run.

Practical default: start with `--draft 1` and the default exact-match verifier. Use `--spec-draft-pq-accept` only when specifically testing stochastic speculative verification.

Use repeated runs before drawing conclusions from a single short run. Speculative acceptance is prompt- and sampler-sensitive, and short generations can swing noticeably.

## Runtime Notes

- `--spec-draft-backend-sampling` exists but is disabled by default for Step MTP. Step's multi-row first pass needs CPU sampling from the final output row; backend top-k sampling is not currently the right path here.
- `--spec-draft-pq-accept` enables experimental stochastic p/q verification for MTP in `llama-server`. The default verifier is exact-match. When p/q is enabled, MTP proposals are sampled from the draft proposal distribution instead of always taking the top-1 token, so the stored draft probability is the actual proposal `q`. Small test matrices have been mixed, so p/q remains opt-in.
- `llama-server` can use the RAM prompt cache with MTP. On prompt-cache restore, the target KV is reused, while the MTP draft context is reset and resumes after fresh target hidden state is produced. This avoids disabling prompt-cache entirely, but the first speculative opportunity after a cache restore may be skipped.
- The MTP draft context runs with embeddings enabled. Warnings about embeddings requiring all input tokens to be marked as outputs are expected for this path.
- `--draft 1` is the default recommendation unless testing a GGUF with more than one trained nextn layer.

## Repeatable Matrix

Run a small fixed-prompt matrix against an already running server:

```bash
python3 scripts/bench-step-mtp.py \
  --url http://127.0.0.1:10009 \
  --draft 1 \
  --temps 0.6,1.0 \
  --pq false,true
```

This prints per-prompt and averaged throughput. If the server response exposes speculative counters, it also prints acceptance; otherwise use the server logs for the acceptance line.
