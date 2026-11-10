# Deterministic gradients: sums in a fixed order

## Abstract

Two trainings from the same seed used to drift apart after a few steps. The cause was the order in which the GPU added floats in the gradient column sums and in the embedding scatter. Both now add in a fixed order, and the same run repeated eight times gives the same gradients.

## What problem it solves (plain language)

If a run cannot be repeated, a speed change cannot be told from a bug: "the loss moved by 2%" may mean a mistake or may mean nothing. Tests that compare two runs then need loose tolerances that hide real defects.

## How it works for a human (intuition, analogy)

Add a long column of prices on a calculator that keeps only seven digits. Top to bottom and bottom to top give totals that differ in the last digit. Each total is fine; they are just not the same. Ask ten people to add parts of the column and shout results in whatever order they finish, and the last digit changes every time.

## Formal view (definitions, formulas, shapes, algorithms)

Float addition is not associative: `(a + b) + c` and `a + (b + c)` can differ by one unit in the last place. `atomicAdd` from many GPU threads adds in the order the threads happen to arrive.

- **Column sums** (bias, LayerNorm gamma and beta, ReLU backward, head bias): each block of threads writes its partial sum to its own slot. A second small kernel adds the partials in runs of consecutive slots, then the run totals in order. Same terms, one fixed association.
- **Embedding scatter:** a token that appears several times in the batch adds several rows into one table row. A first kernel links each token to its next occurrence; a second one walks that chain from the first occurrence, adding in token order.
- **Clip norm:** per-block partials in double precision, added in block order, so the clip factor is the same too.

## Complexity / numerical pitfalls / design choices

- **Why one bit matters.** A last-bit difference in one LayerNorm bias went through Adam, then moved one FFN pre-activation across zero. The ReLU mask flipped, and one step later every gradient differed by about 2%, in about half the runs. Every outcome was a valid gradient of slightly different weights.
- **Cost.** One more small kernel per column sum (LayerNorm backward: 0.089 to 0.103 ms warm). The step gained speed overall in the same change thanks to the [cuBLASLt cache fix](cublaslt-algo-choice.md).
- **Not bit-exact across everything.** The GEMM algorithm is chosen by timing, and algorithms round differently. For a load-independent run also set `ANN_LT_HEURISTIC=1`.
- `ANN_DET_SUM=0` restores the atomics.

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `colsum_reduce_kernel`, `embed_chain_kernel`, `embed_scatter_det_kernel`, `det_sum_on`.
- [`tests/test_gpu.c`](../tests/test_gpu.c): check 9 trains the same model eight times and holds every gradient to 1e-6 of the first run; check 8 does the same for CUDA graphs against the plain path.
- The hunt, with the numbers, is in [CUDA graphs](cuda-graphs-train.md).

## What next

- A few float atomics remain (the loss sum, dQ in the attention backward). Check 9 holds with them; if a gap ever comes back, they are the first place to look.
