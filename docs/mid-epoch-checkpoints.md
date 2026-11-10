# Mid-epoch checkpoints: resume at the same window

## Abstract

With `--ckpt-minutes M` the trainer writes a full checkpoint every `M` minutes, in the middle of an epoch, with the window it reached. `--resume` rebuilds the same shuffle and continues from that window. An epoch that takes days no longer has to survive in one piece.

## What problem it solves (plain language)

Checkpoints used to be written at epoch boundaries only. On a small corpus that is every few minutes; on a large one a single epoch is a multi-day run, and a reboot, a power cut or a crash on day three threw all of it away.

## How it works for a human (intuition, analogy)

A bookmark, plus the guarantee that the book has not been reshuffled. Saving the page number is the easy half. The other half is that the pages come in a random order each epoch, so the resumed run must draw exactly the same order again or the bookmark points at the wrong place.

## Formal view (definitions, formulas, shapes, algorithms)

A checkpoint holds the weights, the Adam state (moments and step count `t`), the epoch number and the window index `idx` inside that epoch.

- **Order:** the epoch's permutation comes from its own generator, seeded by `--seed` and the absolute epoch number. Same seed and same epoch give the same order, in any process.
- **Save:** after an optimizer step, if `M` minutes have passed since the last save, write `(model, Adam, epoch, idx)`.
- **Resume:** load, rebuild the order of that epoch, start the loop at `idx`. Later epochs start at 0.
- **Schedule:** the learning rate depends on Adam's `t`, which is in the checkpoint, so warmup and decay continue where they were (see [LR schedule on continue](lr-schedule-continue.md)).

## Complexity / numerical pitfalls / design choices

- **Never a half-written file.** The checkpoint goes to `PATH.tmp` and is renamed over `PATH`; a crash during the save leaves the previous one intact.
- **At most `M` minutes lost.** A save costs a full write of the weights and both Adam moments, so `M` is a trade between lost work and time spent writing. The long-run driver uses 30.
- **Only at step boundaries.** Gradients of a half-accumulated step are not saved; the step is redone.
- **Same data, same flags.** The window index means something only for the same ids file, `--window`, `--stride` and `--seed`. Change one and the run still starts, on different windows than the bookmark meant.
- **Not bit-identical to an uninterrupted run** unless gradients are deterministic; see [Deterministic gradients](deterministic-gradients.md).
- **Streaming mode** has no window index to save; this applies to the in-memory corpus loop.

## How it shows up in this repo

- [`lang/lm_train.c`](../lang/lm_train.c): `lm_train_set_ckpt_minutes`, `lm_train_set_resume_pos`, `lm_train_set_shuffle_seed`, `lm_train_epoch_order`; the save sits right after the optimizer step and logs `saved checkpoint PATH (epoch E, window I/N)`.
- [`lang/lm_serialize.c`](../lang/lm_serialize.c): `lm_save_checkpoint(model, opt, epoch, window, path)`, written to a temporary file then renamed.
- [`main.c`](../main.c): `--ckpt-minutes M`, `--resume PATH`, `--seed`.
- [`scripts/talk170m_run.sh`](../scripts/talk170m_run.sh): a tmux driver that restarts training with `--resume` after a reboot.

Related: [Checkpoint and resume](checkpoint-and-resume.md).

## Use

```text
anni lm train --data ids.bin --out model.bin --ckpt-minutes 30
# after an interruption
anni lm train --data ids.bin --resume model.bin --ckpt-minutes 30
```
