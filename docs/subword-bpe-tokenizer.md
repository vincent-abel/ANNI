# Subword BPE tokenizer

## Abstract

Byte-pair encoding (BPE) learns a vocabulary of frequent byte sequences by repeatedly merging the most common adjacent pair. The target default is **16k tokens**, with specials at the front.

## What problem it solves

Bytes make sequences long and force the model to rediscover words. Whole-word vocabularies are huge and cannot handle unseen words. BPE sits in between: common words become one token, rare words split into pieces, and nothing is unknown because bytes are the base alphabet.

## How it works for a human

Start with every byte as a token. Count adjacent pairs across the corpus, merge the most frequent pair into a new token, and repeat until the vocab hits the target size. To encode new text, replay the merges in the order they were learned.

## Formal view

Vocabulary \(V_0=\{0..255\}\). For \(t=1..M\): pick \((a,b)=\arg\max \text{count}(a,b)\), set \(V_t=V_{t-1}\cup\{ab\}\), and rewrite the corpus. With specials \(S\), \(|V|=S+256+M\); for 16k, \(M\approx 16000-256-S\). Encoding applies merges by rank: \(O(n\log n)\) with a heap, or \(O(nM)\) naively.

## Complexity / pitfalls / design choices

- **Training cost:** naive pair counting is \(O(M\cdot n)\); sample the corpus or pre-split into words.
- **Determinism:** break count ties by pair order so training is reproducible.
- **Byte fallback** guarantees any UTF-8 round-trips.
- **Embedding table** grows to \(16000\times d\); with \(d=256\) that is about 4M parameters.
- **Trained on train split only** to avoid leaking held-out text.

## How it shows up in this repo

- [`lang/bpe.h`](../lang/bpe.h) / [`lang/bpe.c`](../lang/bpe.c): `bpe_train_from_file` (or `bpe_trainer_*` to feed several pieces), `bpe_encode`, `bpe_decode`, `bpe_save` / `bpe_load`. Default vocab is `BPE_DEFAULT_VOCAB` = 16384; ids are `4` specials, `256` raw bytes (same ids as the byte tokenizer), then one id per merge.
- Pre-tokenization splits text into "words" (optional leading space + a run of letters/digits/high bytes, or of punctuation/newlines; at most 64 bytes) and merges never cross a word. Training counts unique words once, so cost is closer to \(O(M\cdot\text{unique symbols})\) than \(O(M\cdot n)\); pair counts are updated incrementally and the best pair comes from a lazy max-heap. Ties break on the smaller `(left,right)`, so training is deterministic.
- `lm_encode` / `lm_decode` in [`lang/lm.c`](../lang/lm.c) pick the BPE or byte tokenizer for a model; `lm_generate`, `lm_generate_kv`, and the dataset loaders go through them.
- Byte path preserved and still the default: [`lang/tokenize.c`](../lang/tokenize.c), [`lang/vocab.c`](../lang/vocab.c), [Byte tokenizer teaching path](byte-tokenizer-teaching-path.md)
- Merge table stored with weights: [Vocab in checkpoints](vocab-checkpoint-cli.md)
- Vocab file (text, diffable): `BPEV 1 <n_merges>` then one `<left> <right>` id pair per line. `--vocab PATH` on `./ann lm train` trains it if the file is missing; training uses the train split only when `--val-frac` is set.
- Test: `test_bpe_roundtrip` in [`tests/test_ann.c`](../tests/test_ann.c) trains a tiny vocab and checks encode/decode, determinism, and save/load.

## Tiny worked example

Corpus `"aaabdaaabac"`. Most common pair `aa` → `Z`: `ZabdZabac`. Next `ab` → `Y`: `ZYdZYac`. Eleven symbols become seven with two merges.

## What comes next

[Vocab in checkpoints](vocab-checkpoint-cli.md) makes the merge table travel with the model.
