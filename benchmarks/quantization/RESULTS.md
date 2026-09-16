# Initial LLM2Vec quantisation results

These results are a paired sensitivity check, not a release-quality motion
benchmark. The BF16 reference and every quantised variant used the same F32
SOMA SEED denoiser, prompt, 150-frame shape and exact initial-noise bytes. The
prompt is the verified training-split jump exemplar at
`prompts/seed-train/03-jump.txt`.

The sweep used 10 diffusion steps to cover all practical formats on an AMD
Ryzen 9 7900 CPU. Production generation normally uses 100 steps, so absolute
motion-divergence values from this exploratory sweep must not be treated as
acceptance thresholds.

| format | bundle GiB | smaller than BF16 | embedding rel. L2 | embedding cosine | final-state rel. L2 | root mean cm | world MPJPE cm | root-relative MPJPE cm | rotation mean / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 14.14 | - | - | - | - | - | - | - | - |
| Q8_0 | 7.59 | 46.3% | 1.88% | 0.999824 | 1.98% | 0.08 | 0.21 | 0.19 | 0.34° / 1.46° |
| Q6_K | 5.90 | 58.3% | 5.11% | 0.998692 | 13.96% | 0.19 | 2.03 | 1.99 | 3.11° / 16.03° |
| Q5_K | 4.97 | 64.9% | 9.97% | 0.995025 | 17.65% | 0.76 | 2.70 | 2.48 | 3.97° / 15.60° |
| Q4_K | 4.10 | 71.0% | 20.35% | 0.979281 | 21.51% | 1.05 | 2.82 | 2.60 | 4.35° / 17.87° |
| Q4_K_M | 4.72 | 66.6% | 11.31% | 0.993584 | 14.61% | 0.28 | 2.27 | 2.22 | 3.46° / 15.89° |

Q8_0 was close to BF16 on this pair. Q6_K and lower reached a visibly
different pose basin during the jump, which is why more prompts and noise
seeds are required before choosing a default. No non-finite values or runtime
failures were observed.

Uniform Q4_K was anomalously weak. A tensor-level investigation found that
keeping the token/Q/K matrices at Q5_K and V/FFN-down matrices at Q6_K reduced
embedding relative L2 from 20.35% to 11.31% and world MPJPE from 2.82 cm to
2.27 cm. Keeping only the token embedding in BF16 did not improve the mixed
profile (10.84% to 10.88% embedding relative L2 on a second prompt), so the
token table was not the dominant error source. The corrected mixed profile is
available as `q4_k_m`; it remains experimental rather than the recommended
general-purpose format.

## 100-step Q8_0 confirmation

The same prompt and noise were also run at the production 100-step schedule
for BF16 and Q8_0. Q8_0 retained 0.999824 embedding cosine similarity and
produced 0.90 cm mean world MPJPE, 0.60 cm root-relative MPJPE, 0.50 cm mean
root-position error, and 0.76° / 2.41° mean / p95 joint-rotation error. Its
raw final-state relative L2 was 5.28%. This is larger than the 10-step pair but
still geometrically close; a broader prompt/noise sweep is needed to determine
whether it is perceptually interchangeable.

In the paired two-process CPU run, BF16 and Q8_0 sampling both took about
1,179 seconds because the motion denoiser is unchanged. Encoding took 23.84
seconds for BF16 and 8.91 seconds for Q8_0. These concurrent timings are useful
for this machine but are not standalone throughput measurements.

## Second in-distribution exemplar

A second 150-frame, 10-step sweep used the verified training-split dance
prompt (`09-dance.txt`) and seed 43. Its generated artifact can be loaded as a
separate case in the overlay viewer.

| format | embedding rel. L2 | embedding cosine | final-state rel. L2 | root mean cm | world MPJPE cm | root-relative MPJPE cm | rotation mean / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 1.81% | 0.999837 | 2.17% | 0.33 | 0.43 | 0.25 | 0.28° / 1.00° |
| Q6_K | 5.57% | 0.998455 | 5.21% | 1.99 | 2.14 | 0.73 | 0.87° / 2.95° |
| Q5_K | 11.71% | 0.993482 | 52.95% | 26.29 | 28.01 | 8.64 | 8.14° / 27.19° |
| Q4_K | 21.87% | 0.976721 | 21.51% | 13.62 | 14.67 | 4.05 | 5.25° / 16.28° |
| Q4_K_M | 11.70% | 0.993152 | 14.57% | 19.64 | 20.05 | 1.94 | 1.66° / 6.79° |

Q5_K is deliberately reported rather than smoothed away: despite having less
embedding error than Q4_K, it entered a much more distant trajectory basin for
this prompt/noise pair. Replaying the Q5_K sampler from the saved embedding and
noise produced byte-identical final state, root positions, and rotations. The
non-monotonicity is therefore deterministic diffusion sensitivity, not shared
state, changing noise, or a metrics-unit bug. It reinforces the need to judge
formats over several prompt/noise pairs and to inspect root-relative pose error
alongside world-space trajectory drift.

Dance-case identity:

- prompt SHA-256: `b03c5c54b758f89c8ac8246ce4bc285115d87fd0c0ac7ead9118b9055432f7cc`
- noise SHA-256: `74e3bbe95d1c9e4bfabfa431694e369eda9500eed3bdd54183689b5658f9aa87`

Reproduction identity:

- base commit: `568b0253f346fbe369587c7dae73d58594a14c90`
- motion model SHA-256: `14395a62d9c52f40fc63574e62395613760678ecfd0fc136fc73f57f1ff723dc`
- prompt SHA-256: `329a9eb70b1aa40c66fc3d3786f0b39fd6526a127b00b6be9530b508aa8d8c3f`
- noise SHA-256: `7df1dd3e3e32c9b4747152b96f15aab71b582cfdd2984dcabfe572e87b424011`
- backend: CPU; six independent variants, four encoder threads each

Use `scripts/run_quantization_comparison.py` to regenerate the full JSON and
binary animation artifacts. Those large generated files are intentionally
ignored by Git and can be opened at `/compare` in the demo server.
