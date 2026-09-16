# Soliton benchmark: memory predictability

GPU: NVIDIA RTX A6000 · driver 570.86.10 · torch 2.10.0 · jax 0.6.2 · tensorflow 2.20.0

Model: GPT-2 124M, seq 1024, causal attention, AdamW, grad clip 1.0, random tokens. Matmul precision: **tf32** for every framework (TF32 tensor cores allowed).

## E1 · Predict before running, then run

Errors compare the prediction with the framework's own allocator peaks: allocated (live tensors) and reserved (what the allocator holds from the driver, which is what decides OOM).

| framework | batch | predicted GiB | actual alloc / reserved GiB | error vs alloc | error vs reserved | predict time | GPU memory used to predict | NVML peak GiB | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| soliton | 1 | 3.32 | 2.63 / 3.32 | +0.00% | +0.00% | 0.01s | 0.00 | 3.59 | 15,301 |
| soliton | 2 | 4.38 | 3.51 / 4.38 | +0.00% | +0.00% | 0.01s | 0.00 | 4.66 | 20,039 |
| soliton | 4 | 6.34 | 5.41 / 6.34 | +0.00% | +0.00% | 0.01s | 0.00 | 6.61 | 23,373 |
| soliton | 8 | 10.27 | 9.23 / 10.27 | +0.00% | +0.00% | 0.01s | 0.00 | 10.54 | 26,911 |
| soliton | 16 | 18.11 | 16.85 / 18.11 | +0.00% | +0.00% | 0.01s | 0.00 | 18.38 | 28,927 |
| soliton+arena | 1 | 2.60 | 2.60 / 2.60 | +0.00% | +0.00% | 0.38s | 0.00 | 2.90 | 14,742 |
| soliton+arena | 2 | 3.48 | 3.48 / 3.48 | +0.00% | +0.00% | 0.39s | 0.00 | 3.78 | 19,900 |
| soliton+arena | 4 | 5.38 | 5.38 / 5.38 | +0.00% | +0.00% | 0.49s | 0.00 | 5.69 | 23,225 |
| soliton+arena | 8 | 9.19 | 9.19 / 9.19 | +0.00% | +0.00% | 0.45s | 0.00 | 9.50 | 26,818 |
| soliton+arena | 16 | 16.81 | 16.81 / 16.81 | +0.00% | +0.00% | 0.48s | 0.00 | 17.12 | 28,896 |
| pytorch | 1 | 3.10 | 3.13 / 3.39 | -0.93% | -8.61% | 5.45s | 0.26 | 3.71 | 16,844 |
| pytorch | 2 | 4.80 | 4.83 / 5.14 | -0.51% | -6.49% | 5.46s | 0.26 | 5.46 | 20,023 |
| pytorch | 4 | 8.22 | 8.24 / 8.94 | -0.28% | -8.10% | 5.51s | 0.26 | 9.26 | 22,397 |
| pytorch | 8 | 15.04 | 15.06 / 15.83 | -0.13% | -4.97% | 5.48s | 0.26 | 16.15 | 24,159 |
| pytorch | 16 | 28.69 | 28.71 / 31.12 | -0.09% | -7.82% | 5.42s | 0.26 | 31.44 | 25,440 |
| jax | 1 | no API | — / — | — | — | — | — | 8.28 | 24,071 |
| jax | 2 | no API | — / — | — | — | — | — | 12.29 | 26,988 |
| jax | 4 | no API | — / — | — | — | — | — | 20.29 | 29,240 |
| jax | 8 | no API | — / — | — | — | — | — | 35.85 | 31,802 |
| jax | 16 | no API | OOM | — | — | — | — | 16.27 | — |
| tensorflow | 1 | no API | — / 5.44 | — | — | — | — | 8.27 | 9,935 |
| tensorflow | 2 | no API | — / 8.13 | — | — | — | — | 16.27 | 11,680 |
| tensorflow | 4 | no API | — / 12.99 | — | — | — | — | 16.27 | 14,159 |
| tensorflow | 8 | no API | — / 22.90 | — | — | — | — | 32.27 | 15,596 |
| tensorflow | 16 | no API | — / 42.78 | — | — | — | — | 45.91 | 16,354 |

## E2 · Largest batch under a 24.0 GiB budget

| framework | answer from prediction | time, no training runs | correct? | true max (real bisection) | real bisection time | runs / OOMs |
|---|---|---|---|---|---|---|
| soliton | 23 | 2s | yes | 23 | 26s | 7 / 2 |
| soliton+arena | 23 | 7s | yes | 23 | 29s | 7 / 2 |
| pytorch | 13 | 51s | no | 12 | 39s | 7 / 2 |
| jax | no API | — | — | 16 | 250s | 6 / 3 |
| tensorflow | no API | — | — | 8 | 160s | 7 / 4 |

## E3 · Soliton auto-fit under the same 24.0 GiB budget

Soliton may also pick how many blocks to checkpoint (recompute in backward). The choice is made by dry runs only, then verified by training under the hard cap.

| framework | largest batch | blocks checkpointed | planning time, no GPU | trains under cap | peak reserved predicted / actual GiB | tok/s |
|---|---|---|---|---|---|---|
| soliton+autofit | 52 | 13 / 12 | 6s | yes | 23.98 / 23.98 | 22,540 |
