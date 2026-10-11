# Streaming dataloader (E14 mmap + packing)

## Abstract

Wiki-scale `--stream` reads blank-line documents lazily. **E14** memory-maps the corpus when `mmap` succeeds so document scans avoid per-line `fread` syscalls.

## How it works for a human

`text_docs_open` tries `mmap` first, then falls back to `FILE*`. Streaming windows still pack with stride/shuffle as before.

## How it shows up in this repo

- [`lang/dataset_stream.c`](../lang/dataset_stream.c): `text_docs.map`

## What comes next

[Microbatch prefetch](microbatch-prefetch.md) (E3).
