# `--out` and resume confirmation

## Abstract

When `lm train` is given `--out PATH` without `--resume`, and `PATH` already exists, training could silently destroy a checkpoint. The planned behavior warns, offers to resume, and otherwise requires explicit consent. `--force` skips the prompt.

## What problem it solves

Hours of training live in one file. A forgotten flag that overwrites it is an expensive typo. Meanwhile, scripts and CI must never hang on a prompt.

## How it works for a human

If the output file is already there and you did not say `--resume`, the program says so and asks. Answer `y` and it resumes from that file. Anything else overwrites it. With `--force` it overwrites without asking. If there is no terminal (a pipe or CI job), it never prompts and falls back to a safe default.

## Formal view

Let \(E\) = file exists, \(R\) = `--resume` given, \(F\) = `--force`, \(T\) = stdin is a TTY.

- \(\lnot E\) or \(R\): proceed as today.
- \(E \land \lnot R \land F\): overwrite.
- \(E \land \lnot R \land \lnot F \land T\): warn; `y` → resume, else overwrite.
- \(E \land \lnot R \land \lnot F \land \lnot T\): warn on stderr, no prompt, overwrite.

## Complexity / pitfalls / design choices

- **TTY check** via `isatty(0)`; never block a non-interactive run.
- **Default key** is explicit: only `y`/`Y` resumes.
- **Messages go to stderr** so stdout stays parseable.
- **Atomic write:** save to a temp file, then rename, to avoid a half-written checkpoint.

## How it shows up in this repo

- `--out` / `--resume` logic in [`main.c`](../main.c) (currently `--out` defaults to the resume path)
- [Checkpoint and resume](checkpoint-and-resume.md)
- [Train / generate / load CLI](lm-cli.md)

## Tiny worked example

`./ann lm train --out model.bin` with `model.bin` present, on a terminal: "model.bin exists; resume from it? [y/N]". Pressing `y` behaves like `--resume model.bin`. In CI the same command warns and overwrites.

## What comes next

[Byte tokenizer teaching path](byte-tokenizer-teaching-path.md) keeps the simple pipeline alongside subwords.
