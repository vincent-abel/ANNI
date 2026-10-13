# Tiled flash attention (40TF Wave 1)

## Abstract

Under `--flash-attn`, packed MHA stores only per-row **logsumexp** \(L_i\) (size \(B\cdot H\cdot T\)) and never a full \(B\cdot H\cdot T\cdot T\) score tensor in HBM. Backward recomputes \(P_{ij}\) in KV tiles of width `FA_BC` (32) and accumulates \(dQ\), \(dK\), \(dV\) with atomics — FlashAttention-2 style on AnnI’s C/CUDA stack.

## What problem it solves

Materializing attention weights (or recomputing them into a full `w` buffer for bwd) peaks at \(O(B H T^2)\) bytes. On a 16GB card that starves microbatch \(B\) long before Tensor Cores saturate. Tiled bwd caps attention scratch at \(O(B H T)\) plus SRAM tiles.

## How it works

**Forward** (`gpu_attn_flash_fwd`): same causal row kernel as before, with `w = NULL`, and write

\[
L_i = m_i + \log \ell_i,\quad m_i=\max_j S_{ij},\ \ell_i=\sum_j e^{S_{ij}-m_i}.
\]

**Backward** (`gpu_attn_tiled_bwd`): for each query row \(i\), \(D_i=\langle dO_i, O_i\rangle\). For each key tile \(j\in[j_0,j_0+B_c)\):

\[
P_{ij}=e^{S_{ij}-L_i},\quad
dS_{ij}=P_{ij}(dP_{ij}-D_i)\cdot\mathrm{scale},\quad
dP_{ij}=\langle dO_i,V_j\rangle.
\]

Accumulate \(dQ_i += dS_{ij} K_j\), atomic \(dK_j += dS_{ij} Q_i\), atomic \(dV_j += P_{ij}\,dO_i\).

**Wire:** `fused==4` in [`lang/attn.c`](../lang/attn.c). `lm_packed_bytes` counts \(B\cdot H\cdot T\) floats for flash, not \(T\times T\).

## Formal complexity

| Path | HBM peak (attn scores) |
|------|-------------------------|
| Fused / TC | \(B H T^2\) |
| Legacy flash bwd (recompute `w`) | \(B H T^2\) peak |
| Tiled flash | \(B H T\) (`L`) + tile SRAM |

Compute remains \(O(B H T^2 d_h)\) per layer; the win is **VRAM → larger \(B\)** → higher tok/s on the train meter.

## How it shows up in this repo

- CLI: `--flash-attn` (prefers tiled over TC when fwd succeeds)
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `g_fa_L`, `attn_tiled_bwd_kernel`, `gpu_attn_tiled_bwd`
- Cap: **T ≤ 256** (`ATT_BLOCK`)
- Related: [flash-attention.md](flash-attention.md), [sota-train-systems.md](sota-train-systems.md)

## Pitfalls

- Atomics on \(dK\)/\(dV\) are correct but not peak-bandwidth; a later FA3-style producer/consumer tiling can replace them.
- `L` must come from the matching forward; graph replay that skips CE is fine, but dtype/shape change must recapture.
- TC GemmEx path still needs `w_all` when tiled fwd fails — keep for small-B experiments.
