# LM weight serialization

## Abstract

[`network_save`](../net/serialize.c) writes **ANN1** magic and a fully-connected **layer graph**. Language-model checkpoints are a **flat tensor bundle** (config + embed + blocks + head) with a **distinct on-disk format**—planned magic **`LMCK`** (four bytes) or **`LM1\0`**—so loaders never confuse an LM with an XOR network. API: **`lm_save`**, **`lm_load`**.

## What problem it solves (plain language)

Training for many epochs on a corpus is wasted if weights vanish when the process exits. Serialization **freezes** every learned matrix to disk so you can **resume**, evaluate, or **generate** later without retraining from random init.

## How it works for a human (intuition, analogy)

A checkpoint is a **photograph** of all dials in the model at one moment: not the training code, not the dataset—just numbers arranged in a agreed order so another program can reconstruct the same machine.

## Formal view (definitions, formulas, shapes, algorithms)

**File layout (little-endian doubles, planned).**

| Offset field | Type | Value |
|--------------|------|-------|
| magic | char[4] | `'L','M','C','K'` |
| version | uint32 | 1 |
| d_model | uint32 | 32 |
| n_blocks | uint32 | 2 |
| n_heads | uint32 | 4 |
| d_ff | uint32 | 64 |
| max_ctx | uint32 | 32 |
| vocab_size | uint32 | 256 |
| n_tensors | uint32 | count |
| payload | n_tensors × (tag + dims + raw doubles) | see below |

**Tensor walk order (must match load).**

1. `embed` — `(V, d)`
2. For each block: `wq, wk, wv, wo` — `(d,d)`; `w1` `(d,d_ff)`; `b1` `(d_ff)`; `w2` `(d_ff,d)`; `b2` `(d)`; `ln1_gamma, ln1_beta`; `ln2_gamma, ln2_beta` — `(d)`
3. `w_head` `(d,V)`, `b_head` `(V)`

Optional **Adam state** extension in version 2 ([checkpoint-and-resume.md](checkpoint-and-resume.md)): append `t`, then per-slot `m`, `v` in same slot order as [`adam-lm-weights.md`](adam-lm-weights.md).

**Not ANN1.** ANN1 files begin with different magic and encode `n_layers`, neuron counts—`lm_load` must **reject** ANN1 with clear error.

**Planned API (`lang/lm_io.h`).**

```c
int  lm_save(const lm *model, const char *path);
lm  *lm_load(const char *path);   /* lm_create + fill weights */
int  lm_save_adam(const lm *model, const lm_adam *opt, const char *path);
lm  *lm_load_adam(const char *path, lm_adam **opt_out);
```

**Algorithm `lm_save`.**

```text
open binary write
write header + config constants (cross-check lm_config.h)
for each tensor in walk order:
  write dims, write data[i] row-major
close
```

**Algorithm `lm_load`.**

```text
read magic; if not LMCK -> error
read version, dims; assert match compile-time LM_* or build compatible model
allocate lm; fread each tensor into model->...
return model
```

## Complexity / numerical pitfalls / design choices (CS depth)

**Portability.** Fixed-width integers in header; doubles IEEE-754—fine Linux↔Windows if both little-endian.

**Config mismatch.** Loading a file trained with `d_model=64` into code compiled with 32 must **fail fast** at header check ([scale-toward-target-size.md](scale-toward-target-size.md)).

**Partial files.** Truncated read → error, free partial `lm`.

**Security.** Loading untrusted binaries is arbitrary code execution only if buffer overflows—use exact size checks from header × sizeof(double).

**Human readability.** Optional sidecar JSON for config only; binary stays canonical for size.

## How it shows up in this repo

- Reference: [`docs/model-serialization.md`](model-serialization.md) — ANN1 walk for `network`.
- LM weights today: **not persisted**; only in-memory `lm_create`.
- Planned: `lang/lm_io.c`, used by CLI `lm load` / train `--out` ([lm-cli.md](lm-cli.md), [generation-from-checkpoints.md](generation-from-checkpoints.md)).

## Worked example

Save after training to `checkpoints/toy.bin`. File starts with bytes `4C 4D 43 4B` (`LMCK`). Payload includes first embed row (256 doubles for first token’s vector if stored row-major). Reload: `lm *m = lm_load("checkpoints/toy.bin"); lm_nll(m, ids, n)` matches pre-save within floating noise.

## What next

- Implement v1 tensor-only format; add v2 Adam tail.
- Test round-trip: random weights → save → load → byte-identical tensors.
- Document default checkpoint path in CLI help.
