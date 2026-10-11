# LR schedule, grad clip, weight decay (E2)

## Abstract

CLI warmup + cosine/linear decay, optional grad clip and AdamW-style weight decay so Target-scale FP32 runs stay stable without hand-tuning every epoch.

## Flags

- `--lr-warmup N`
- `--lr-decay cosine|linear|none` with `--lr-decay-steps M`
- `--grad-clip MAX_NORM` (host Adam path)
- `--weight-decay WD`

## How it shows up in this repo

- [`lang/lm_train.c`](../lang/lm_train.c): `train_step_lr`
- [`lang/lm_adam.c`](../lang/lm_adam.c): clip + wd on host step
- [`main.c`](../main.c)

## What comes next

[Tokens per step](tokens-per-step.md) (E9).
