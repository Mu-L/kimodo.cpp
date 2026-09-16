# Text-encoder quantisation benchmark

This benchmark treats the BF16 native text bundle as the paired reference and
changes only the LLM2Vec base-weight representation. Every compared motion
uses the same F32 Kimodo denoiser, exact initial-noise file, frame count,
diffusion schedule and CFG weights.

`prompts/seed-train/manifest.json` contains auditable BONES-SEED training-split
exemplars. See [`docs/QUANTIZATION.md`](../../docs/QUANTIZATION.md) for bundle
generation, reporting and overlay-viewer commands. The first controlled CPU
sweep is recorded in [`RESULTS.md`](RESULTS.md).

Do not interpret paired geometric divergence as a standalone motion-quality
score. A quantised result can differ from the BF16 sample while remaining a
valid realization of the prompt. Use multiple prompts/noises and NVIDIA's
distribution-level motion benchmark before setting a release threshold.
