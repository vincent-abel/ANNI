# LR schedule for continued training

## Abstract

`--lr-sched-from STEP` makes warmup and decay count from a given Adam step instead of from zero. It is what lets a finished model train some more with a fresh warmup and a fresh cosine.

## What problem it solves (plain language)

The learning rate follows Adam's step counter, and that counter is saved in the checkpoint. A model that finished its run sits at the end of its schedule. Resume it for a second phase with a cosine and the learning rate is zero from the first step.

## How it works for a human (intuition, analogy)

A stopwatch that keeps the total time, with a lap button. The total (Adam's step) must keep running because Adam's own math uses it. The schedule reads the lap time.

## Formal view (definitions, formulas, shapes, algorithms)

With `t` the Adam step, `s0` the offset, `w` the warmup steps, `n` the decay steps:

```text
s  = max(0, t - s0)
lr = base * (s + 1) / w                       if s < w
p  = min(1, (s - w) / n)
lr = base * (1 - p)                           linear
lr = base * 0.5 * (1 + cos(pi * p))           cosine
```

With `s0 = 0` this is the schedule of [LR schedule, clip, weight decay](lr-schedule-clip-wd.md).

## Complexity / numerical pitfalls / design choices

- **Why not reset Adam's step?** Its bias correction and its moments belong to the step count. Resetting it would blow the first updates up.
- **The offset is not saved.** Pass the same `--lr-sched-from` again when resuming the second phase.
- **A constant LR needs no offset.** The chat fine-tunes use `--lr-decay none`.
- The control tower's manual LR still wins over the schedule ([Training control tower](training-control-tower.md)).

## How it shows up in this repo

- [`lang/lm_train.c`](../lang/lm_train.c): `lm_train_step_lr(base_lr, step)` and `lm_train_set_lr_sched_from(step)`.
- `anni lm train --lr-sched-from STEP` in [`main.c`](../main.c).
- `test_lr_sched_from` in [`tests/test_ann.c`](../tests/test_ann.c): warmup, cosine, and the clamp before the offset.

## Worked example

talk-170m ended pretraining at Adam step 29084. Its stretch to 2048 tokens ([2048-token context](context-2048.md)) ran 3052 more steps with:

```text
--lr 1e-4 --lr-warmup 100 --lr-decay cosine --lr-decay-steps 3052 --lr-sched-from 29084
```

Step 29084 is the first warmup step, step 29184 is at the full 1e-4, and the cosine ends where the data ends.
