# LLM2Vec quantisation and divergence workflow

Kimodo conditions motion generation on a pooled 4096-value LLM2Vec embedding.
The tools here quantise only the large BF16 token/projection matrices. RMSNorm
weights and the supervised LoRA branch remain BF16/F32 respectively. This
keeps the small, numerically sensitive tensors out of the first quantisation
experiment.

## Build quantised bundles

After building the project and obtaining the BF16 text bundle:

```sh
build/debug/kmd-quantize-text generated/llm2vec-text-bundle \
  generated/llm2vec-text-q8_0 q8_0
build/debug/kmd-quantize-text generated/llm2vec-text-bundle \
  generated/llm2vec-text-q6_k q6_k
build/debug/kmd-quantize-text generated/llm2vec-text-bundle \
  generated/llm2vec-text-q5_k q5_k
build/debug/kmd-quantize-text generated/llm2vec-text-bundle \
  generated/llm2vec-text-q4_k q4_k
build/debug/kmd-quantize-text generated/llm2vec-text-bundle \
  generated/llm2vec-text-q4_k_m q4_k_m
```

The destination must be absent. Each GGUF records `kimodo.quantization` and
the GGML quantisation version. Quantisation happens offline; inference never
silently rewrites weights. The tool stages the entire bundle beside the final
path, removes that staging directory on an ordinary failure, and publishes it
with one rename only after every component succeeds.

The uniform levels are useful for measuring sensitivity. `q4_k_m` is the
preferred low-bit experiment: it uses Q4_K for most base matrices, Q5_K for
the token/Q/K matrices, and Q6_K for the more sensitive V and FFN-down
projections. Norm and LoRA tensors remain unquantised in every profile.

## Pack release GGUFs

The runtime accepts the component directories above for compatibility, but the
published format stores one seekable weight GGUF per quantization and one
shared tokenizer:

```sh
mkdir -p generated/llm2vec-text-packed
build/debug/kmd-pack-text generated/llm2vec-text-bundle \
  generated/llm2vec-text-packed/Llama-3-Kimodo-BF16.gguf
build/debug/kmd-pack-text generated/llm2vec-text-q8_0 \
  generated/llm2vec-text-packed/Llama-3-Kimodo-Q8_0.gguf
# Repeat for Q6_K, Q5_K, Q4_K, and Q4_K_M.
ln -s ../llm2vec-text-bundle/tokenizer.gguf \
  generated/llm2vec-text-packed/tokenizer.gguf
```

Packing streams tensor payloads and does not materialize the model in RAM. At
inference time Kimodo parses the catalog once and seeks directly to the
embedding, selected layer, or final norm. A monolith therefore preserves
bounded layer streaming and supports full residency without 34 weight files.

## Select verified in-distribution prompts

Download NVIDIA's `train_split_paths.txt` and SEED timeline annotations, then
run:

```sh
python scripts/select_seed_exemplars.py \
  --train-split /path/to/train_split_paths.txt \
  --timelines /path/to/timelines.jsonl \
  --output benchmarks/quantization/prompts/seed-train
```

The selector only accepts annotations whose exact filename occurs in the
published BONES-SEED training split. Its manifest records source hashes, split
entries, event timestamps and prompt text. This supports the SOMA/G1 SEED
models. The proprietary Rigplay training corpus cannot be verified this way;
official demo prompts for RP models must be labelled as curated rather than
confirmed training-distribution examples.

## Produce a controlled comparison

The comparison runner encodes each prompt independently, writes or copies one
initial-noise tensor, and reuses that exact tensor for every variant:

```sh
python scripts/run_quantization_comparison.py \
  --motion-model models/kimodo-soma-seed-v1.1-f32.gguf \
  --skeleton soma30 \
  --prompt benchmarks/quantization/prompts/seed-train/03-jump.txt \
  --reference bf16=generated/llm2vec-text-packed/Llama-3-Kimodo-BF16.gguf \
  --variant q8_0=generated/llm2vec-text-packed/Llama-3-Kimodo-Q8_0.gguf \
  --variant q6_k=generated/llm2vec-text-packed/Llama-3-Kimodo-Q6_K.gguf \
  --variant q5_k=generated/llm2vec-text-packed/Llama-3-Kimodo-Q5_K.gguf \
  --variant q4_k=generated/llm2vec-text-packed/Llama-3-Kimodo-Q4_K.gguf \
  --variant q4_k_m=generated/llm2vec-text-packed/Llama-3-Kimodo-Q4_K_M.gguf \
  --frames 150 --steps 100 --seed 42 \
  --jobs 3 --threads 8 \
  --output quantization-comparisons/seed-jump-42
```

For cross-runtime comparisons, pass `--noise captured.f32`; matching a numeric
seed alone does not guarantee the same normal samples. `--jobs` runs
independent variants concurrently; the encoder still processes each text
independently, preserving upstream's batch-size-one behaviour.

The bundled results were validated on CPU because no Vulkan device was
available in the measurement environment. Use `--backend vulkan` only after
checking the quantised encoder on the target driver/device.

The report contains:

- embedding cosine/angular error, relative L2, norm ratio, RMSE and maximum
  absolute error;
- raw final diffusion-state divergence;
- root trajectory and endpoint error;
- world and root-relative MPJPE;
- sign-invariant quaternion geodesic error; and
- joint-velocity error plus per-frame viewer series.

Run several captured noises per prompt for aggregate conclusions. A paired
divergence report measures sensitivity, not semantic quality; use NVIDIA's
motion benchmark separately for text alignment, R-precision and FID.

## View overlays

```sh
go run ./demo -addr 0.0.0.0:8094 \
  --comparisons quantization-comparisons
```

Open `http://localhost:8094/compare`. The BF16 and quantised skeletons share a
frame clock and camera. World-space mode exposes trajectory drift;
root-position-aligned mode isolates pose differences. Variants can be hidden
and their opacity adjusted, while per-frame MPJPE and aggregate rotation error
are shown beside the animation.

Initial CPU measurements and their reproduction hashes are recorded in
[`benchmarks/quantization/RESULTS.md`](../benchmarks/quantization/RESULTS.md).
