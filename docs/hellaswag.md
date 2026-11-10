# HellaSwag bench

## Abstract

HellaSwag gives a context and four possible endings; only one makes sense. `anni lm hellaswag` scores a checkpoint on the 10,042 validation questions, and `scripts/run555m_checks.py` runs it at each step of the 555M run, into the HellaSwag tab of the dashboard.

## What problem it solves (plain language)

The training loss only compares a model with itself. HellaSwag is a public test, so a score can be compared across steps and with other models.

## How it works for a human (intuition, analogy)

The model is not asked to answer. It reads each of the four endings after the context, and the ending it finds the least surprising is its choice.

## Formal view (definitions, formulas, shapes, algorithms)

For ending `e` of `b` bytes, with `L(e)` the sum of the token losses of `e` given the context:

- **acc**: `argmin L(e)` is the right ending
- **acc_norm**: `argmin L(e) / b` is the right ending, the usual headline number
- **nll_token**: mean token loss of the right endings

Zero-shot; chance is 25%. Context and ending are tokenized apart, the ending with its leading space, and the context loses its start if the sum passes the trained window of 256. Sequences are sorted by length and run 32 at a time, padded to a multiple of 32 tokens: attention is causal, so padding cannot change the losses read. Without a GPU the same scores are computed one sequence at a time, which only suits `--limit`.

## Why not the letter format

[deepeval](https://deepeval.com/docs/benchmarks-hellaswag) shows the endings as A to D, with examples, and wants the letter back. That tests following a multiple-choice format, which a small base model cannot do: it would sit at chance at every step. Log-likelihood is how base models are normally reported, and it moves from early in training.

## Use

```text
scripts/hellaswag_data.py        # once, needs pyarrow: data/hellaswag_val.tsv
scripts/hellaswag_build.sh       # once per code change: build/hellaswag/anni
build/hellaswag/anni lm hellaswag --model MODEL --vocab data/mix100k.vocab [--limit N]
```

It prints one JSON line. Results of the run are in `logs/run555m/hellaswag.jsonl`.
