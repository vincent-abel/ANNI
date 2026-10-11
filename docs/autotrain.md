# Autotrain executable

## Abstract

`./autotrain` is a second binary beside `./ann`: an opinionated, one-command path that probes the GPU, ensures training data, builds or loads a BPE vocab, picks window/batch/lr/epochs, runs packed training, then smoke-generates. Optional `--sweep` runs a short LR grid (AT7–AT9) with early-stop and diverge ditch before the full budget.

## What problem it solves

Reaching the README Target needs many flags (BPE, stream, val split, batch auto, safe LR). Newcomers and overnight runs benefit from a single entry point that chooses sane defaults and writes a reproducible plan file next to the checkpoint.

## How it works for a human

```text
make autotrain          # or: make cuda-autotrain
./autotrain --out wiki.bin [--data PATH] [--epochs N] [--sweep]
```

It prints a short plan (data size, B, T, lr, epochs, VRAM), trains with stream + `--batch auto` defaults, writes `OUT.autotrain.json`, then smoke-generates. With `--sweep`, it tries a few learning rates for one epoch each, keeps the best by val metric (E15/E16), then continues.

## Formal view

Phase 1 (AT1–AT6) maps environment \(\mathcal{E}\) (VRAM free, data path) to a config \(c = (B,T,\eta,E,\ldots)\) and runs the packed trainer \(f(c)\). Phase 2 (AT7–AT9) searches a small set \(C\) for \(\arg\min_{c\in C} \mathrm{NLL}_{\mathrm{val}}(c)\) on a short horizon, then continues from that checkpoint with plateau early-stop and diverge abort.

## Complexity / pitfalls / design choices

- **Separate exec:** keeps `ann` pedagogic; autotrain calls library APIs (not `system("./ann")`).
- **Defaults favor Target plumbing:** BPE + stream + val-frac + lr `0.001` + varlen (E8).
- **Resume:** if `--out` is already `LMCF`/`LMCV`, continue rather than silently re-init.
- **Repro:** write `OUT.autotrain.json` with the resolved flags (E11).
- **Not a full hyperparameter optimizer:** phase 2 is a tiny LR grid, not Bayesian search.

## How it shows up in this repo

- [`autotrain.c`](../autotrain.c), `make autotrain` / `make cuda-autotrain` — **AT1–AT9**
- Roadmap checklist: **AT1–AT9** in [`../README.md`](../README.md) under **Autotrain**
- Reuses [`lang/lm_train.h`](../lang/lm_train.h), [`lang/bpe.h`](../lang/bpe.h), [`scripts/fetch_wiki.sh`](../scripts/fetch_wiki.sh)

| Id | Behavior |
| --- | --- |
| AT1 | `make autotrain` builds `./autotrain` |
| AT2 | BPE, stream, val-frac, batch auto, lr 0.001 |
| AT3 | Missing `--data` → fetch wiki or exit offline |
| AT4 | Vocab beside `--out`; resume when present |
| AT5 | Plan print + `OUT.autotrain.json` |
| AT6 | Smoke generate + final metric |
| AT7 | `--sweep` short `{lr}` grid |
| AT8 | Continue best; E15 early-stop |
| AT9 | E16 diverge ditch on sweep + full train |

## Tiny worked example

```text
$ make autotrain && ./autotrain --out wiki.bin --sweep
plan: data=data/wiki_simple.txt … B=auto T=64 lr=1e-3
sweep lr=0.0003 …
AT7 best sweep idx 1 lr=0.001
… train …
smoke: The history of ...
autotrain done → wiki.bin
```

## What comes next

Richer batch sweeps, Target-quality overnight recipes, and wiring best-ckpt restore paths more aggressively after diverge.
