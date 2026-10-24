# Talking model plan (talk-170m)

## Abstract

Train the current Target shape (\(D=1024\), \(L=12\), \(D_{FF}=4096\), 16k BPE, about 185M parameters) on roughly 3B tokens of mixed web text and code, then fine-tune it on chat transcripts in the `User:` / `Assistant:` format that `cli/chat_session.c` already speaks. The goal is a model that holds a short conversation and writes toy code. Everything up to the GPU run is CPU work; the GPU phase is one long pretraining run (about one to two days on the 5070 Ti) plus a short fine-tune.

## What problem it solves

The only real corpus so far is Simple English Wikipedia: about 105 MB, about 25M tokens. That is about 1% of what a model this size needs, so a D=1024 run on it mostly memorises. A model also does not learn to answer questions from encyclopedia prose alone: it needs conversations, and it needs code if it should ever write code.

## Formal view

Parameters at \(V=16384\) with an untied head:

\[
N = L\bigl(4D^2 + 2D\,D_{FF}\bigr) + 2VD
= 12\,(4.19 + 8.39)\,\mathrm{M} + 2 \cdot 16.8\,\mathrm{M}
\approx 151\,\mathrm{M} + 34\,\mathrm{M} \approx 185\,\mathrm{M}.
\]

The compute-optimal budget of Hoffmann et al. (2022) is about 20 tokens per parameter:

\[
T \approx 20N \approx 3.7\times 10^9 \text{ tokens}.
\]

Training time follows from the measured throughput (see [Scale D=1024](scale-d1024-40tf.md) and `.cursor/agent-sync.md`, 2026-10-18): about 60–68 TF at \(P_{\mathrm{meter}} \approx 252\mathrm{M}\), which is

\[
\mathrm{tok/s} \approx \frac{60\times 10^{12}}{6 \cdot 2.52\times 10^{8}} \approx 40\,\mathrm{k}.
\]

That figure was measured with the byte vocab at \(T=64\). The 16k head and a longer window both cost throughput, so the plan assumes 25–40k tok/s:

| Tokens | at 40k tok/s | at 25k tok/s |
|---|---|---|
| 3.0B | 21 h | 33 h |
| 3.7B | 26 h | 41 h |

## Data mix

| Source | Role | License | Share | Tokens |
|---|---|---|---|---|
| FineWeb-Edu `sample/10BT` (Penedo et al., 2024) | English prose, educational filter | ODC-By 1.0 | ~85% | ~2.6B |
| codeparrot/github-code-clean, Python, C, C++, JavaScript | code | rows kept only under MIT, Apache-2.0, BSD, ISC | ~15% | ~0.4B |
| SmolTalk (Allal et al., 2025), OpenAssistant oasst2 | chat fine-tune only | Apache-2.0 | separate stage | ~0.1B |

`scripts/fetch_mix.py` downloads the parquet files over HTTPS (no login) into `data/raw/` and writes `data/mix_web.txt`, `data/mix_code.txt` and held-out `data/mix_val_*.txt` (every 1000th document of each source). All of it is gitignored, and `data/mix.SOURCE` records URLs, licenses and counts.

FineWeb-Edu files hold about 5.2 GB of text each, about 1.25B tokens, so the run caps web text at pretokenization instead of downloading fewer files.

## Tokenizer

Keep the 16k BPE size: a larger vocab grows the head (\(V\cdot D\)) and the logits buffer that already limits batch size at 16k. Retrain the merges on a sample of the mix in the same 85/15 proportion, because `model-wiki-16k.bin.vocab` was learned on Wikipedia only and splits code badly. Measured on this machine: BPE training takes about 48 s per 50 MB of text, and encoding runs at about 30 MB/s (about 4.2 bytes per token on English).

The pre-tokenizer splits text into words of one byte class with an optional leading space. Every extra space is its own word, so code indentation costs one token per space. This is noted as a known cost, not changed in this step, because changing it would re-tokenize existing checkpoints differently.

## Pipeline

1. **Fetch** — `scripts/fetch_mix.py` (CPU, network).
2. **Tokenizer** — train 16k BPE on a mixed sample (CPU).
3. **Pretokenize** — `scripts/pretok_ids` per source, then concatenate into `data/mix_train_ids.bin` and `data/mix_val_ids.bin` (int32 LE, about 12 GB for 3B tokens). `text_dataset` now counts ids in a `long`, so one file may exceed \(2^{31}\) ids; use `--stride` equal to `--window` so window indices stay below `INT_MAX`.
4. **Pretrain** (GPU) — one epoch over the train ids.
5. **Chat fine-tune** (GPU) — SmolTalk and oasst2 rendered as `User:` / `Assistant:` turns, a few hundred million tokens at a lower learning rate.

## Checkpoints and evaluation

- Save every epoch, plus a `.best` on held-out NLL (existing behaviour).
- Held-out loss on `mix_val_ids.bin`, reported separately for web and code.
- Fixed generation prompts after each checkpoint: one prose prompt, one Python function stub, and one `User:` question.
- Known risk: the 2026-10-18 Wikipedia run at `--lr 3e-4` diverged. Use warmup and a lower peak LR, and keep the diverge ditch on.

## How it shows up in this repo

- [`scripts/fetch_mix.py`](../scripts/fetch_mix.py): data fetch
- [`lang/dataset_text.c`](../lang/dataset_text.c): `n_ids` is a `long`; window count clamps at `INT_MAX`
- [`.gitignore`](../.gitignore): `data/raw/`, `data/mix_*`
- Next notes: tokenizer on the mix, pretokenized mix, chat fine-tune data

## References

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022.
- Penedo et al., *The FineWeb Datasets*, 2024.
- Allal et al., *SmolLM2: When Smol Goes Big*, 2025.
- Eldan and Li, *TinyStories*, 2023.
