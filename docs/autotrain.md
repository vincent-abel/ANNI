# Autotrain executable

## Abstract

`./autotrain` is a planned **second binary** beside `./ann`: an opinionated, one-command path that probes the GPU, ensures training data, builds or loads a BPE vocab, picks window/batch/lr/epochs, runs packed training, then smoke-generates. The teaching CLI (`./ann lm train …`) stays fully explicit.

## What problem it solves

Reaching the README Target needs many flags (BPE, stream, val split, batch auto, safe LR). Newcomers and long overnight runs benefit from a single entry point that chooses sane defaults and writes a reproducible plan file next to the checkpoint.

## How it works for a human

You run `./autotrain` (optionally with `--out` / `--data`). It prints a short plan (data size, B, T, lr, epochs, VRAM), trains, then shows a few generated tokens and final NLL. Later, a light tune phase can try a few learning rates for a short budget and keep the best before a longer train.

## Formal view

Phase 1 maps environment \(\mathcal{E}\) (VRAM free, data path) to a config \(c = (B,T,\eta,E,\ldots)\) and runs the existing packed trainer \(f(c)\). Phase 2 searches a small set \(C\) for \(\arg\min_{c\in C} \mathrm{NLL}_{\mathrm{val}}(c)\) on a short horizon, then continues from that checkpoint.

## Complexity / pitfalls / design choices

- **Separate exec:** keeps `ann` pedagogic; autotrain can call `system` for wiki fetch or refuse offline.
- **Defaults favor Target plumbing:** BPE + stream + val-frac, not byte toy defaults.
- **Resume:** if `--out` is already `LMCF`/`LMCV`, continue rather than silently re-init.
- **Repro:** write `OUT.autotrain.json` with the resolved flags (ties to efficiency item E11).
- **Not a full hyperparameter optimizer:** phase 2 is a tiny grid, not Bayesian search.

## How it shows up in this repo

- [`autotrain.c`](../autotrain.c), `make autotrain` / `make cuda-autotrain` — **AT1–AT6 MVP** (BPE stream train, resume, plan JSON, smoke generate)
- Roadmap checklist: **AT1–AT9** in [`../README.md`](../README.md) under **Autotrain**
- Reuses train/pack APIs in [`lang/lm_train.h`](../lang/lm_train.h), BPE in [`lang/bpe.h`](../lang/bpe.h), fetch in [`scripts/fetch_wiki.sh`](../scripts/fetch_wiki.sh)

## Tiny worked example

```text
$ make autotrain && ./autotrain --out wiki.bin
plan: data=data/wiki_simple.txt (120 MB) B=auto→96 T=64 lr=1e-3 epochs=1
... train ...
smoke: The history of ...
train NLL 3.21  val window NLL 3.45
```

## What comes next

Implement AT1–AT6 (MVP), then AT7–AT9 with plateau early-stop (**E15**) and diverge ditch (**E16** / **AT9**): if NLL spikes or goes non-finite, abort that trial and restore the best checkpoint. Efficiency stack R1 / huge B / graphs / mixed precision makes autotrain wall time practical at Target scale.
