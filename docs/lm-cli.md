# LM CLI

## Abstract

The **`./ann`** binary already runs a **demo LM path** (`run_lm` in [`main.c`](../main.c)): create model, train on a **phrase** with `lm_train_steps`, print NLL before/after, **`lm_generate`** a short continuation. The planned **`ann lm`** subcommand tree formalizes **train**, **generate**, **load**, and **eval** over corpus files, checkpoints, and hyperparameters—without changing the XOR `network` commands.

## What problem it solves (plain language)

Researchers and students need one entry point to **train on a file**, **save weights**, **resume**, **measure perplexity**, and **sample text**—not recompile different `main` variants for each experiment.

## How it works for a human (intuition, analogy)

Like `git` subcommands (`git commit`, `git push`), **`ann lm`** groups language-model actions under one noun so flags stay predictable and help text stays scoped.

## Formal view (definitions, formulas, shapes, algorithms)

**Current behavior (shipped).**

```text
./ann lm [<phrase>]
  default phrase: "hello"
  srand(1); lm_create()
  loss0 = lm_nll(ids); loss1 = lm_train_steps(model, phrase, 80, 0.05)
  lm_generate(model, phrase, out, 8, tau=0.0)
  print NLL and generation
```

**Target syntax.**

```text
ann lm train   [--corpus PATH] [--epochs N] [--batch B] [--lr ETA]
               [--checkpoint PATH] [--resume PATH] [--out PATH]

ann lm generate [--checkpoint PATH] --prompt TEXT [--max N] [--temp TAU]

ann lm load     --checkpoint PATH [--prompt TEXT]  /* inspect NLL optional */

ann lm eval     --checkpoint PATH --corpus PATH   /* perplexity */
```

**Dispatch (planned `main.c` structure).**

```text
if argc >= 2 && strcmp(argv[1], "lm") == 0:
  if argc >= 3 && strcmp(argv[2], "train") == 0:
    run_lm_train(parse_flags(...))
  else if ... "generate" -> run_lm_generate
  else if ... "eval"     -> run_lm_eval
  else:
    run_lm_demo(argv)   /* backward compatible phrase mode */
```

**`run_lm_train` algorithm.**

```text
model = resume ? lm_load_adam(ckpt) : lm_create()
opt = lm_adam_create(model)
ds = text_dataset_load(corpus default "data/toy.txt")
loop epochs: lm_train_corpus(model, opt, ds, 1, batch, lr)
lm_save_adam(model, opt, out_path)
print lm_perplexity_corpus on holdout if configured
```

**Flags defaults (proposed).**

| Flag | Default |
|------|---------|
| corpus | `data/toy.txt` |
| epochs | 10 |
| batch | 4 |
| lr | 0.001 |
| out | `checkpoints/lm.bin` |
| max (gen) | 64 |
| temp | 0 (greedy) |

**Exit codes.** 0 success; 1 alloc/IO/arg error; non-zero eval if PPL above threshold (tests only).

## Complexity / numerical pitfalls / design choices (CS depth)

**Arg parsing.** Minimal hand-rolled parser vs getopt—match existing `main.c` style (simple loops).

**Relative paths.** Resolve corpus/checkpoint relative to **cwd**, not executable path—document for WSL users.

**Backward compatibility.** Bare `./ann lm hello` keeps phrase demo until corpus trainer is default.

**Build.** No separate binary; link all `lang/*.o` already in Makefile (do not edit Makefile in doc-only tasks—implementation follows).

**Help.** `./ann lm help` prints subcommands linking to these docs.

## How it shows up in this repo

- [`main.c`](../main.c): `run_lm`, top-level argv routing.
- [`main.h`](../main.h): declarations as needed.
- LM API surface: [`lang/lm.h`](../lang/lm.h), planned `lm_train_corpus`, `lm_load`, `lm_perplexity` ([train-all-parameters.md](train-all-parameters.md), [lm-weight-serialization.md](lm-weight-serialization.md)).
- Generation: [`lang/generate.h`](../lang/generate.h).

## Worked example

```bash
# Phrase demo (today)
./ann lm "hello world"

# Planned corpus train + generate
./ann lm train --corpus data/toy.txt --epochs 20 --batch 8 --out /tmp/toy.bin
./ann lm eval --checkpoint /tmp/toy.bin --corpus data/toy_holdout.txt
./ann lm generate --checkpoint /tmp/toy.bin --prompt "Once " --max 40 --temp 0.9
```

Expected stdout: epoch losses, final train/holdout PPL, then a generated string appended to prompt.

## What next

- Implement flag parsing and subcommand dispatch without breaking XOR modes.
- Wire train/eval to `text_dataset` + `lm_train_corpus`.
- Add `ann lm help` and mention defaults in [`README.md`](../README.md) when CLI stabilizes (README edits separate from this note).
