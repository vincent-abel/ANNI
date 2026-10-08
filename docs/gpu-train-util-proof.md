# nvidia-smi / train util proof

## Abstract

[`./ann lm train --bench-gpu`](../main.c) runs synthetic train microsteps via [`lm_bench_train_ms`](../lang/lm_bench.c) and prints **backend**, **gpu_available**, tokens, steps, and mean ms. [`lm bench --train`](../main.c) is the same loop under the bench subcommand. Document how to corroborate with `nvidia-smi dmon`.

## What problem it solves (plain language)

Forward-only `lm bench` lights up the GPU; full train used to look idle. This CLI proves the train path is exercising the linked backend and tells you how to watch util.

## How it works for a human (intuition, analogy)

A stopwatch run of a few weightlifting sets (forward+backward+Adam), then a label on the gym door saying which machine you used—and a tip to glance at the power meter (`nvidia-smi`).

## Formal view (definitions, formulas, shapes, algorithms)

Each step: zero grads → `lm_forward_cache` → `lm_loss_backward` → `lm_backward` → `lm_adam_step`.

Report: \(\mathrm{mean\_ms} = 1000\cdot\Delta t / n_{\mathrm{steps}}\).

## Complexity / numerical pitfalls / design choices

**Synthetic ids.** No corpus I/O; focuses on compute and PCIe from the train kernels.

**CPU stub.** `available=0` and `backend=cpu` on default `make`; rebuild with `make cuda` for device util.

## How it shows up in this repo

- [`lang/lm_bench.c`](../lang/lm_bench.c): `lm_bench_train_ms`
- [`main.c`](../main.c): `lm train --bench-gpu`, `lm bench --train`

## Worked example

```bash
make cuda
./ann lm train --bench-gpu --tokens 32 --steps 30
# other terminal:
nvidia-smi dmon -s u -c 20
```

Expect non-zero util while steps run (teaching kernels or cuBLAS). On CPU builds the CLI still prints timings and reminds you to `make cuda`.

## What next

Roadmap complete for GPU training checklist; further work is multi-GPU / FP16 (out of scope here).
