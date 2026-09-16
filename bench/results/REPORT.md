# Soliton benchmark: memory predictability

GPU: NVIDIA RTX A6000 · driver 570.86.10 · torch 2.10.0 · jax 0.6.2 · tensorflow 2.20.0

Model: GPT-2 124M, seq 1024, fp32 (TF32 off), naive attention, AdamW, grad clip 1.0, random tokens.

## E1 · Predict before running, then run

Errors compare the prediction with the framework's own allocator peaks: allocated (live tensors) and reserved (what the allocator holds from the driver, which is what decides OOM).

| framework | batch | predicted GiB | actual alloc / reserved GiB | error vs alloc | error vs reserved | predict time | GPU memory used to predict | NVML peak GiB | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| soliton | 1 | 4.14 | 3.36 / 4.14 | +0.00% | +0.00% | 0.05s | 0.00 | 4.41 | 8,161 |
| soliton | 2 | 5.86 | 4.78 / 5.86 | +0.00% | +0.00% | 0.29s | 0.00 | 6.13 | 10,255 |
| soliton | 4 | 9.12 | 7.81 / 9.12 | +0.00% | +0.00% | 0.03s | 0.00 | 9.39 | 11,925 |
| soliton | 8 | 15.64 | 13.87 / 15.64 | +0.00% | +0.00% | 0.03s | 0.00 | 15.91 | 12,967 |
| soliton | 16 | 28.67 | 25.99 / 28.67 | +0.00% | +0.00% | 0.03s | 0.00 | 28.93 | 13,302 |
| pytorch | 1 | 3.10 | 3.13 / 3.39 | -0.93% | -8.61% | 5.38s | 0.26 | 3.71 | 11,303 |
| pytorch | 2 | 4.80 | 4.83 / 5.14 | -0.51% | -6.49% | 5.29s | 0.26 | 5.46 | 13,334 |
| pytorch | 4 | 8.22 | 8.24 / 8.94 | -0.28% | -8.10% | 5.28s | 0.26 | 9.26 | 14,566 |
| pytorch | 8 | 15.04 | 15.06 / 15.83 | -0.13% | -4.97% | 5.40s | 0.26 | 16.15 | 15,481 |
| pytorch | 16 | 28.69 | 28.71 / 31.12 | -0.09% | -7.82% | 5.41s | 0.26 | 31.44 | 16,012 |
| jax | 1 | no API | — / — | — | — | — | — | 8.28 | 20,704 |
| jax | 2 | no API | — / — | — | — | — | — | 12.29 | 24,501 |
| jax | 4 | no API | — / — | — | — | — | — | 20.29 | 26,429 |
| jax | 8 | no API | — / — | — | — | — | — | 35.85 | 28,234 |
| jax | 16 | no API | OOM | — | — | — | — | 16.27 | — |
| tensorflow | 1 | no API | — / 5.56 | — | — | — | — | 8.27 | 7,811 |
| tensorflow | 2 | no API | — / 8.23 | — | — | — | — | 16.27 | 8,868 |
| tensorflow | 4 | no API | — / 12.97 | — | — | — | — | 16.27 | 10,460 |
| tensorflow | 8 | no API | — / 22.98 | — | — | — | — | 32.27 | 11,453 |
| tensorflow | 16 | no API | — / 42.61 | — | — | — | — | 45.91 | 11,891 |

## E2 · Largest batch under a 24.0 GiB budget

| framework | answer from prediction | time, no training runs | correct? | true max (real bisection) | real bisection time | runs / OOMs |
|---|---|---|---|---|---|---|
| soliton | 14 | 2s | yes | 14 | 32s | 7 / 4 |
| pytorch | 13 | 52s | no | 12 | 43s | 7 / 2 |
| jax | no API | — | — | 15 | 323s | 7 / 3 |
| tensorflow | no API | — | — | 8 | 173s | 7 / 4 |

## E3 · Soliton auto-fit under the same 24.0 GiB budget

Soliton may also pick how many blocks to checkpoint (recompute in backward). The choice is made by dry runs only, then verified by training under the hard cap.

| framework | largest batch | blocks checkpointed | planning time, no GPU | trains under cap | peak reserved predicted / actual GiB | tok/s |
|---|---|---|---|---|---|---|
| soliton+autofit | 52 | 12 / 12 | 4s | yes | 23.87 / 23.87 | 9,914 |
