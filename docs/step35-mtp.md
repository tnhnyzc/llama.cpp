# Step 3.5 Flash MTP

This branch is based on StepFun's Step 3.5 MTP llama.cpp fork:

- upstream fork: https://github.com/stepfun-ai/llama.cpp
- upstream branch: `step3p5-mtp`
- local base commit: `a2f5ec441` (`recover requires_dft lost in rebase`)

The StepFun branch added Step 3.5 Flash runtime/model-loading support, MTP speculative support, SWA KV rollback, prompt-cache plumbing, and MTP quantization/conversion support. This branch layers local server/runtime hardening on top: prompt-cache restore behavior for MTP, SWA prompt-cache threshold fixes, MTP prompt-cache tail hidden-row handling, optional server-side p/q acceptance, a Step thinking-template flag, and local benchmarking notes.

The implementation is experimental same-GGUF MTP support for Step 3.5 Flash models that contain `step35.nextn_predict_layers`.

The locally tested path is:

```bash
./build/bin/llama-server \
  -m /path/to/Step-3.5-Flash-MTP-IQ4_XS-3.90BPW-Q8_MTP.gguf \
  -mtp \
  --draft 1 \
  -np 1 \
  -ngl 99
```

## Current Recommendation

Use `-mtp --draft 1` for the current public Step 3.5 Flash MTP GGUF tested here.

On the local Apple M3 Max test setup, a short server run improved from about `28.4 tok/s` without MTP to about `34.6 tok/s` with `-mtp --draft 1`. A broader four-prompt `temp 0.6` matrix found the same shape: `--draft 1` was the best average setting, while deeper drafts accepted more total draft tokens but did not pay for their extra MTP and verification work.

A later controlled 384-token check on the `IQ3_S-3.64BPW-Q8_MTP` quant showed:

| Runtime | Prompt cache path | Decode speed | Draft acceptance |
| --- | --- | ---: | ---: |
| This fork, no MTP | default cache path | 26.87 t/s | - |
| This fork, `-mtp --draft 1` | default cache path | 32.55 t/s | 168/214, 78.5% |
| StepFun `step3p5-mtp`, `-mtp --draft 1` | default cache path | 27.10 t/s | disabled by prompt-cache path |
| StepFun `step3p5-mtp`, `-mtp --draft 1 --cache-ram 0` | prompt cache disabled | 33.51 t/s | 168/214, 78.5% |

That comparison is the practical reason for the local prompt-cache/MTP handling:
the StepFun branch can run MTP, but its server prompt-cache path disables MTP in
this scenario. This fork resets the MTP draft-side state after prompt-cache
restore and resumes speculation once fresh target hidden state is available.

The tested GGUF reports:

```text
step35.nextn_predict_layers = 1
```

## Local GGUFs

The locally prepared GGUFs use cleaner public names:

| File | Notes |
| --- | --- |
| `Step-3.5-Flash-MTP-IQ4_XS-3.90BPW-Q8_MTP.gguf` | AesSedai-style mixed expert layout; MTP/nextn tensors kept Q8. |
| `Step-3.5-Flash-MTP-IQ3_S-3.64BPW-Q8_MTP.gguf` | Smaller custom IQ3_S expert layout; MTP/nextn tensors kept Q8. |
| `Step-3.5-Flash-MTP-IQ3_XXS-3.27BPW-Q8_MTP.gguf` | Smallest custom IQ3_XXS expert layout; MTP/nextn tensors kept Q8. |

The imatrix used for these local quantizations came from Bartowski's Step 3.5 Flash GGUF work, not from this fork. Credit it separately when publishing model cards.

So `--draft 2` and deeper drafts reuse the single MTP layer recurrently. They are not true multi-head MTP for this model file. The runtime now maps draft step `k` to nextn layer `base + k` when a future Step GGUF exposes multiple nextn layers, while clamping to the last available layer if the requested draft depth is larger than the model supports.

In one four-prompt matrix, using request-level dotted keys `"speculative.n_max"` and `"speculative.pq_accept"` at `temp 0.6`, the averages were:

| `n_max` | p/q accept | avg tok/s | avg acceptance |
| --- | --- | ---: | ---: |
| 1 | off | 29.85 | 0.780 |
| 1 | on | 30.49 | 0.763 |
| 2 | off | 27.04 | 0.663 |
| 2 | on | 28.01 | 0.664 |
| 3 | off | 27.07 | 0.644 |
| 3 | on | 26.96 | 0.645 |
| 4 | off | 26.37 | 0.643 |
| 4 | on | 25.40 | 0.617 |

This is why `--draft 3` and `--draft 4` are not recommended for the currently tested one-nextn GGUF. They are useful diagnostics, but not a faster runtime path here.

A later matrix after prompt-cache fixes showed that p/q is workload-sensitive rather than a universal win:

| temperature | p/q accept | avg tok/s | avg acceptance |
| --- | --- | ---: | ---: |
| 0.6 | off | 38.17 | 0.783 |
| 0.6 | on | 32.64 | 0.759 |
| 1.0 | off | 29.60 | 0.700 |
| 1.0 | on | 30.33 | 0.789 |

So the current practical split is:

- `temp 0.6`, `--draft 1`, exact-match verifier for the fast default path;
- `temp 1.0`, `--draft 1`, `--spec-draft-pq-accept` when testing Xiaomi-style stochastic sampling.

Use repeated runs before drawing conclusions from a single chat session. Speculative acceptance is prompt- and sampler-sensitive, and short generations can swing noticeably.

## Runtime Notes

- `--spec-draft-backend-sampling` exists but is disabled by default for Step MTP. Step's multi-row first pass needs CPU sampling from the final output row; backend top-k sampling is not currently the right path here.
- `--spec-draft-pq-accept` enables experimental stochastic p/q verification for MTP in `llama-server`. The default verifier is exact-match. When p/q is enabled, MTP proposals are sampled from the draft proposal distribution instead of always taking the top-1 token, so the stored draft probability is the actual proposal `q`. Local matrices have been mixed: p/q helped `temp 1.0` acceptance, but did not consistently beat exact-match at `temp 0.6`. It remains opt-in.
- `llama-server` can use the RAM prompt cache with MTP. On prompt-cache restore, the target KV is reused, while the MTP draft context is reset and resumes after fresh target hidden state is produced. This avoids disabling prompt-cache entirely, but the first speculative opportunity after a cache restore may be skipped.
- The MTP draft context runs with embeddings enabled. Warnings about embeddings requiring all input tokens to be marked as outputs are expected for this path.
- `--draft 1` is the default recommendation unless testing a GGUF with more than one trained nextn layer.

## Repeatable Local Matrix

Run a small fixed-prompt matrix against an already running server:

```bash
python3 scripts/bench-step-mtp.py \
  --url http://127.0.0.1:10009 \
  --draft 1 \
  --temps 0.6,1.0 \
  --pq false,true
```

This prints per-prompt and averaged throughput. If the server response exposes speculative counters, it also prints acceptance; otherwise use the server logs for the acceptance line.

## Notes From MiMo MTP Work

The Step findings line up with the broader MoE MTP work:

- Acceptance rate matters, but it is not enough by itself. Extra verification rows and MTP rows still have to be cheap enough to pay for themselves.
- Low-depth MTP is the conservative runtime path when deeper trained heads are absent or when verifier cost dominates.
- Exact-match verification is conservative for stochastic sampling. p/q acceptance is the cleaner experimental path for non-greedy generation, but it is not automatically faster unless the extra accepted tokens offset verifier and sampler cost. The current implementation uses the draft top-k proposal distribution available from the MTP sampler.
- Backend sampling and tree-style schedulers are useful only when their graph shape matches the model path. Blindly offloading sampling or increasing draft depth can make the system slower.
- Quantization of MTP tensors can affect acceptance. For experimental MTP quants, keep nextn tensors as high precision as practical unless benchmarks show the quantized heads preserve acceptance.
