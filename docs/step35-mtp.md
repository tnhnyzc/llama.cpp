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

On the local Apple M3 Max test setup, a short server run improved from about `28.4 tok/s` without MTP to about `34.6 tok/s` with `-mtp --draft 1`. The same prompt with `--draft 2` was slower, around `31.8 tok/s`, despite accepting more total draft tokens.

The tested GGUF reports:

```text
step35.nextn_predict_layers = 1
```

So `--draft 2` and deeper drafts reuse the single MTP layer recurrently. They are not true multi-head MTP for this model file. The runtime now maps draft step `k` to nextn layer `base + k` when a future Step GGUF exposes multiple nextn layers, while clamping to the last available layer if the requested draft depth is larger than the model supports.

## Runtime Notes

- `--spec-draft-backend-sampling` exists but is disabled by default for Step MTP. Step's multi-row first pass needs CPU sampling from the final output row; backend top-k sampling is not currently the right path here.
- `--spec-draft-pq-accept` enables experimental stochastic p/q verification for MTP. The default verifier is exact-match. In one local `temp 0.6` smoke test it did not help `--draft 1`, but improved `--draft 2` from about `29.4 tok/s` to about `35.3 tok/s`. That still did not beat the tested `--draft 1` path, so p/q remains opt-in.
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
