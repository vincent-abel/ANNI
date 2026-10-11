# Throughput dashboard (E10)

## Abstract

Train progress now reports tokens/sec, a rough TFLOP/s estimate, and VRAM used alongside ms/tok so efficiency work is measurable without a separate profiler.

## What problem it solves

ms/tok alone hides whether the GPU is filled. Tokens/sec and VRAM high-water show whether huge-B and residency changes actually moved the needle.

## How it works for a human

Watch the `-d` progress line: `ms/t`, `t/s`, `~TF`, and `MB`. The tower shows the same fields. Estimates are coarse; use `nvidia-smi` for util%.

## Formal view

\(\mathrm{tok/s}=N_{\mathrm{tok}}/t_{\mathrm{wall}}\). Rough train FLOPs \(\approx 6\,P\,N_{\mathrm{tok}}\) for a dense decoder with \(P\) non-embed parameters → TFLOP/s \(=6 P\cdot(\mathrm{tok/s})/10^{12}\).

## Complexity / pitfalls / design choices

- \(P\) omits embed/head for a quick bound; good for relative comparisons.
- VRAM is `total-free` at sample time, not a true peak allocator watermark.

## How it shows up in this repo

- [`lang/train_progress.h`](../lang/train_progress.h): `tok_sec`, `tflops_est`, `vram_used_mb`
- [`lang/lm_train.c`](../lang/lm_train.c): `fill_throughput`, `-d` progress line
- [`cli/tower.c`](../cli/tower.c): tower speed row

## Tiny worked example

10k tokens in 1.0 s → 10k t/s. If \(P\approx 3\cdot10^6\), TFLOP/s \(\approx 6\cdot3e6\cdot1e4/1e12=0.18\).

## What comes next

[Compile profile docs](compile-profile.md) (E12); pair with R-stack benches in README.
