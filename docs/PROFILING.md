# Inference profiling

Set `KIMODO_PROFILE=1` to print phase timings for text-component uploads,
embedding, transformer execution, motion-weight upload, graph construction,
graph allocation, host/device transfers, GPU compute, diffusion steps, and
decode. GGML's lower-level Vulkan counters remain available through
`GGML_VK_PERF_LOGGER=1`.

The measurements below were taken on an NVIDIA GeForce RTX 5070 Ti (16 GiB)
with the 19-token jump exemplar, the Q8_0 text bundle, and a 60-frame SOMA
SEED sample. Times vary with clocks and shader-cache state.

| Path | Before | Current | Change |
| --- | ---: | ---: | ---: |
| Repeated Q8 text encode | 1,253 ms | 67 ms isolated; 75 ms beside resident motion weights | about 17–19x faster |
| 60-frame motion diffusion step | 67.3 ms | 50.9 ms | 24% faster |
| Motion graph calls per step | 36 | 12 | 3x fewer |

The Q8 encoder's one-time upload takes about 1.1–1.2 seconds. The live worker
keeps that encoder and the selected motion checkpoint in VRAM, so requests
using the same model/quantization avoid subsequent weight uploads. On this
workstation the resulting native worker occupies about 8,894 MiB of VRAM.
Changing either selector replaces the worker and therefore incurs a new cold
load.

The BF16 bundle is 14,481 MiB before motion weights and inference buffers. It
cannot safely remain resident with the motion model on a 16 GiB GPU. The demo
therefore applies `KIMODO_TEXT_RESIDENT_LIMIT_MIB=10000`: Q8 and smaller
bundles stay resident, while BF16 streams eight-layer groups. The measured BF16
reference encode was 5.79 seconds in this safe mode.

## Operation breakdown

For a warm Q8 text encode, Vulkan timestamps reported about 49 ms of GPU work.
The main quantized FFN projections dominate. The rank-16 F32 LoRA down
projections are the next largest group and run at low utilization because of
their narrow shape. Sharing the attention normalization for Q, K, and V removes
two redundant RMS-normalization subgraphs per layer without changing output
bytes.

Before motion-layer grouping, one 60-frame diffusion step spent approximately:

- 38.5 ms in `ggml_backend_graph_compute`
- 8.4 ms allocating graph buffers
- 3.4 ms uploading inputs
- 5.2 ms downloading intermediate outputs

Executing four transformer layers per graph removes most intermediate host
round-trips and allocator calls. Output is byte-identical to the one-layer
graph path; set `KIMODO_MOTION_LAYER_CHUNK=1` to reproduce that baseline or use
the default value of 4.

GGML's per-operation motion trace identifies the next optimization targets:

1. Packed attention. The parity-first graph launches 768 small QK products and
   768 small value products per diffusion step, plus 3,074 `CONT` operations.
   A packed layout should remove much of this launch and materialization cost,
   but its numerical motion divergence must be measured before becoming the
   default.
2. Reusable graph allocations. Twelve graphs are still built and allocated on
   every diffusion step. Shape-keyed graph/buffer caching can remove much of
   the remaining allocator overhead.
3. Larger motion graph groups. More than four layers needs a larger GGML graph
   capacity and activation-memory measurements; four is the validated default.
4. LoRA fusion for the text encoder. Fusing the rank-16 update with each base
   projection would target the remaining narrow F32 matrix multiplications.

For repeatable text measurements, `kmd-encode` accepts an optional repetition
count and reports every iteration:

```sh
KIMODO_BACKEND=vulkan KIMODO_TEXT_LAYER_CHUNK=32 KIMODO_PROFILE=1 \
  build/debug/kmd-encode generated/llm2vec-text-q8_0 \
  benchmarks/quantization/prompts/seed-train/03-jump.txt /tmp/jump.f32 3
```
