# Training control tower

## What it is

An optional **ncurses** full-screen UI for LM training and a companion **inspect** view for any checkpoint. Build enables it when `ncurses.h` is present (`ANN_NCURSES` + `-lncurses` in the Makefile). Without headers, `./ann lm train -d` still works; tower commands tell you to install `libncurses-dev`.

## Commands

```bash
# Live train UI (same flags as lm train)
./ann lm tower train --data data/toy_tiny.txt --epochs 3 --out model.bin --force

# Full-screen intel on a checkpoint
./ann lm tower inspect --model model.bin --prompt "The " [--data PATH]

# Plain-text intel (no ncurses; good for scripts)
./ann lm info --model model.bin --prompt hello [--data PATH] [--eval-windows M]
```

## Train panes

| Pane | Content |
|------|---------|
| Header | data/out paths, arch, vocab, parameter count |
| Progress | epoch / window bars, NLL, win NLL, val NLL, ms/tok, ETA |
| GPU | `gpu_backend_name()` + VRAM free/total when CUDA is linked |
| Loss | ASCII sparkline of recent win/val NLL |
| Log | ring of epoch / save / control messages |
| Sample | last `lm_generate` output |
| Footer | key help |

## Keys (train)

- **Space** — pause / resume (cooperative between microbatches)
- **+** / **-** — multiply / divide learning rate by 1.5
- **g** — generate ~32 tokens from `--prompt` (default `The `)
- **s** — save checkpoint now to `--out`
- **q** — quit after the current step (checkpoint already saved each epoch; mid-epoch save if `s` was used)

## Inspect / info

Reports file size, magic (`LMCF` / `LMCV` / legacy `LMCK`), `d` / heads / ff / ctx / blocks, tokenizer, **parameter count**, weight bytes, checkpoint epoch/step/adam_t, GPU memory, optional prompt NLL/PPL and corpus window NLL.

## How it hooks the train loop

[`lang/train_progress.h`](../lang/train_progress.h) exposes:

- `lm_train_set_progress` — callback after each microbatch / epoch / save
- `lm_train_set_control` — `pause`, `quit_save`, `save_now`, mutable `lr`

[`lang/lm_train.c`](../lang/lm_train.c) still uses stderr `-d` progress when no callback is set. The tower attaches both hooks for the duration of `tower_train_begin` … `tower_train_end`.

## Files

- [`cli/tower.c`](../cli/tower.c) / [`cli/tower.h`](../cli/tower.h) — UI
- [`lang/lm_info.c`](../lang/lm_info.c) — shared intel fill/print
- [`lang/train_progress.h`](../lang/train_progress.h) — progress/control API
