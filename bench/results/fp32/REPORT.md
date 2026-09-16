# Soliton benchmark: memory predictability

GPU: NVIDIA RTX A6000 · driver 570.86.10 · torch 2.10.0 · jax 0.6.2 · tensorflow 2.20.0

Model: GPT-2 124M, seq 1024, causal attention, AdamW, grad clip 1.0, random tokens. Matmul precision: **fp32** for every framework (true fp32, tensor cores off).

## E1 · Predict before running, then run

Errors compare the prediction with the framework's own allocator peaks: allocated (live tensors) and reserved (what the allocator holds from the driver, which is what decides OOM).

| framework | batch | predicted GiB | actual alloc / reserved GiB | error vs alloc | error vs reserved | predict time | GPU memory used to predict | NVML peak GiB | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| soliton | 1 | 3.32 | 2.63 / 3.32 | +0.00% | +0.00% | 0.01s | 0.00 | 3.59 | 10,052 |
| soliton | 2 | 4.38 | 3.51 / 4.38 | +0.00% | +0.00% | 0.01s | 0.00 | 4.66 | 13,095 |
| soliton | 4 | 6.34 | 5.41 / 6.34 | +0.00% | +0.00% | 0.01s | 0.00 | 6.61 | 14,926 |
| soliton | 8 | 10.27 | 9.23 / 10.27 | +0.00% | +0.00% | 0.01s | 0.00 | 10.54 | 16,678 |
| soliton | 16 | 18.11 | 16.85 / 18.11 | +0.00% | +0.00% | 0.01s | 0.00 | 18.38 | 17,759 |
| soliton+arena | 1 | 2.60 | 2.60 / 2.60 | +0.00% | +0.00% | 0.62s | 0.00 | 2.90 | 10,001 |
| soliton+arena | 2 | 3.48 | 3.48 / 3.48 | +0.00% | +0.00% | 0.48s | 0.00 | 3.78 | 13,117 |
| soliton+arena | 4 | 5.38 | 5.38 / 5.38 | +0.00% | +0.00% | 0.68s | 0.00 | 5.69 | 14,903 |
| soliton+arena | 8 | 9.19 | 9.19 / 9.19 | +0.00% | +0.00% | 0.43s | 0.00 | 9.50 | 16,714 |
| soliton+arena | 16 | 16.81 | 16.81 / 16.81 | +0.00% | +0.00% | 0.45s | 0.00 | 17.12 | 17,661 |
| pytorch | 1 | 3.10 | 3.13 / 3.39 | -0.93% | -8.61% | 5.31s | 0.26 | 3.71 | 11,281 |
| pytorch | 2 | 4.80 | 4.83 / 5.14 | -0.51% | -6.49% | 5.36s | 0.26 | 5.46 | 13,212 |
| pytorch | 4 | 8.22 | 8.24 / 8.94 | -0.28% | -8.10% | 5.33s | 0.26 | 9.26 | 14,618 |
| pytorch | 8 | 15.04 | 15.06 / 15.83 | -0.13% | -4.97% | 5.45s | 0.26 | 16.15 | 15,576 |
| pytorch | 16 | 28.69 | 28.71 / 31.12 | -0.09% | -7.82% | 5.50s | 0.26 | 31.44 | 16,005 |
| jax | 1 | no API | — / — | — | — | — | — | 6.28 | 14,038 |
| jax | 2 | no API | — / — | — | — | — | — | 10.28 | 15,803 |
| jax | 4 | no API | — / — | — | — | — | — | 18.28 | 17,009 |
| jax | 8 | no API | — / — | — | — | — | — | 34.28 | 18,037 |
| jax | 16 | no API | — / — | — | — | — | — | 34.28 | 18,196 |
| tensorflow | 1 | no API | — / 5.84 | — | — | — | — | 8.27 | 7,776 |
| tensorflow | 2 | no API | — / 8.12 | — | — | — | — | 16.27 | 7,976 |
| tensorflow | 4 | no API | — / 13.00 | — | — | — | — | 16.27 | 10,464 |
| tensorflow | 8 | no API | — / 22.92 | — | — | — | — | 32.27 | 11,447 |
| tensorflow | 16 | no API | — / 42.70 | — | — | — | — | 45.91 | 11,892 |

## E2 · Largest batch under a 24.0 GiB budget

| framework | answer from prediction | time, no training runs | correct? | true max (real bisection) | real bisection time | runs / OOMs |
|---|---|---|---|---|---|---|
| soliton | 23 | 2s | yes | 23 | 30s | 7 / 2 |
| soliton+arena | 23 | 7s | yes | 23 | 34s | 7 / 2 |
| pytorch | 13 | 51s | no | 12 | 43s | 7 / 2 |
| jax | no API | — | — | 16 | 187s | 6 / 3 |
| tensorflow | no API | — | — | 8 | 183s | 7 / 4 |

## E3 · Soliton auto-fit under the same 24.0 GiB budget

Soliton may also pick how many blocks to checkpoint (recompute in backward). The choice is made by dry runs only, then verified by training under the hard cap.

| framework | largest batch | blocks checkpointed | planning time, no GPU | trains under cap | peak reserved predicted / actual GiB | tok/s |
|---|---|---|---|---|---|---|
| soliton+autofit | 52 | 13 / 12 | 6s | yes | 23.98 / 23.98 | 14,165 |
