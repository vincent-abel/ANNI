# Checkpoint and resume

## Abstract

Long corpus training should survive **interruption**. A **checkpoint** bundles **model weights** ([lm-weight-serialization.md](lm-weight-serialization.md)) plus **Adam state** (`m`, `v`, timestep `t` per slot from [`adam-lm-weights.md`](adam-lm-weights.md)) and optional **training metadata** (epoch, dataset cursor, rng seed). **Resume** reloads all three so optimization continues as if the process had not stopped.

## What problem it solves (plain language)

Adam is not memoryless: its adaptive steps depend on **past gradient statistics**. Saving only weights and restarting Adam from zero changes the effective learning dynamics. Saving **optimizer state** plus **epoch/cursor** avoids repeating data or overshooting training budget.

## How it works for a human (intuition, analogy)

Checkpointing is **saving a video game**: character stats (weights), experience bars (Adam m/v), and **which level you were on** (epoch + file cursor)—not just a screenshot of the character.

## Formal view (definitions, formulas, shapes, algorithms)

**Checkpoint contents (logical).**

- `model`: all parameters θ
- `optimizer`: `{ t, {(m_i, v_i) for each slot i} }`
- `trainer`: `{ epoch, cursor, seed, last_loss }`

**On-disk.** Version 2 LMCK file: header + tensors + `uint64 t` + `2 × n_slots` doubles (m then v interleaved per slot, or block of all m then all v—document one choice).

**Resume algorithm.**

```text
model, opt = lm_load_adam(path)
ds = text_dataset_load("data/toy.txt")
ds->cursor = saved_cursor   /* if stored in sidecar or extended header */
for epoch in saved_epoch .. target_epochs:
  train as lm_train_corpus with existing opt->t
```

**Fresh vs resume Adam `t`.** Continue `t` from checkpoint so bias correction \(\hat{m}, \hat{v}\) stays consistent:

\[
\hat{m} = \frac{m}{1-\beta_1^t}
\]

Resetting `t=0` while keeping large `m,v` **mis-scales** updates.

**Sidecar JSON (optional).**

```json
{ "epoch": 3, "cursor": 1204, "corpus": "data/toy.txt", "lr": 0.001 }
```

**Atomic write.** Write `path.tmp`, `fsync`, rename to `path` to avoid corrupt half files.

## Complexity / numerical pitfalls / design choices (CS depth)

**Size.** Adam doubles parameter count on disk (~2× model-only checkpoint)—still <1 MB for toy LM.

**Cursor + stride.** After resume, first window must match **next unseen** window; store cursor, not epoch only.

**Changing batch_size on resume.** Allowed if grad accumulation semantics unchanged; changing lr is fine; changing **corpus file** invalidates compare.

**RNG.** Store `srand` seed for reproducible shuffle if added later.

**Version skew.** Loading v1 (no Adam) + creating fresh `lm_adam` is **warm restart**, not true resume—log warning.

## How it shows up in this repo

- Net precedent: no optimizer in ANN1 saves—LM training **adds** optimizer persistence first-class.
- CLI: `./ann lm train --data data/toy.txt --resume model.bin --epochs 5` ([lm-cli.md](lm-cli.md)).
- Functions: `lm_save_adam`, `lm_load_adam` ([lm-weight-serialization.md](lm-weight-serialization.md)).

## Worked example

Train 2 epochs on toy.txt, save at end of epoch 1 with `cursor=500`, `t=1250`. Kill process. Resume: load checkpoint, verify `opt->t == 1250`, run epoch 2 starting at cursor 500—total windows processed equals uninterrupted 2-epoch run (within fp drift).

## What next

- Extend binary format v2 with Adam payload.
- Optional: store dataset cursor in checkpoint metadata for mid-epoch resume.
- Integration test: train N steps, save, load, train M steps vs train N+M continuous.
