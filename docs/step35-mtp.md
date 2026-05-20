# Step 3.5 Flash MTP

This fork includes experimental same-GGUF MTP support for Step 3.5 Flash models that contain `step35.nextn_predict_layers`.

The locally tested path is:

```bash
./build/bin/llama-server \
  -m /path/to/Step-3.5-Flash-MTP.gguf \
  -mtp \
  --draft 1 \
  -np 1 \
  -ngl 99
```

## Current Recommendation

Use `-mtp --draft 1` for the current public Step 3.5 Flash MTP GGUF tested here.

On the local Apple M3 Max test setup, a short server run improved from about `28.4 tok/s` without MTP to about `34.6 tok/s` with `-mtp --draft 1`. A broader four-prompt `temp 0.6` matrix found the same shape: `--draft 1` was the best average setting, while deeper drafts accepted more total draft tokens but did not pay for their extra MTP and verification work.

The tested GGUF reports:

```text
step35.nextn_predict_layers = 1
```

So `--draft 2` and deeper drafts reuse the single MTP layer recurrently. They are not true multi-head MTP for this model file. The runtime now maps draft step `k` to nextn layer `base + k` when a future Step GGUF exposes multiple nextn layers, while clamping to the last available layer if the requested draft depth is larger than the model supports.

In the four-prompt matrix, using request-level dotted keys `"speculative.n_max"` and `"speculative.pq_accept"` at `temp 0.6`, the averages were:

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

## Runtime Notes

- `--spec-draft-backend-sampling` exists but is disabled by default for Step MTP. Step's multi-row first pass needs CPU sampling from the final output row; backend top-k sampling is not currently the right path here.
- `--spec-draft-pq-accept` enables experimental stochastic p/q verification for MTP in `llama-server`. The default verifier is exact-match. When p/q is enabled, MTP proposals are sampled from the draft proposal distribution instead of always taking the top-1 token, so the stored draft probability is the actual proposal `q`. In the local four-prompt `temp 0.6` matrix, p/q slightly improved average throughput for `--draft 1` and `--draft 2`, but did not make deeper recurrent drafts beat `--draft 1`. It remains opt-in.
- `llama-server` disables prompt cache automatically when MTP is active. Prompt-cache reuse would require reconstructing the target hidden states used to seed the MTP first pass, which is not implemented yet.
- The MTP draft context runs with embeddings enabled. Warnings about embeddings requiring all input tokens to be marked as outputs are expected for this path.
- `--draft 1` is the default recommendation unless testing a GGUF with more than one trained nextn layer.

## Notes From MiMo MTP Work

The Step findings line up with the broader MoE MTP work:

- Acceptance rate matters, but it is not enough by itself. Extra verification rows and MTP rows still have to be cheap enough to pay for themselves.
- Low-depth MTP is the conservative production path when deeper trained heads are absent or when verifier cost dominates.
- Exact-match verification is conservative for stochastic sampling. p/q acceptance is the cleaner experimental path for non-greedy generation, but it is not automatically faster unless the extra accepted tokens offset verifier and sampler cost. The current implementation uses the draft top-k proposal distribution available from the MTP sampler.
- Backend sampling and tree-style schedulers are useful only when their graph shape matches the model path. Blindly offloading sampling or increasing draft depth can make the system slower.
- Quantization of MTP tensors can affect acceptance. For experimental MTP quants, keep nextn tensors as high precision as practical unless benchmarks show the quantized heads preserve acceptance.
