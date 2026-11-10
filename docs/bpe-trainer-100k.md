# BPE trainer at scale: the 100k vocabulary

## Abstract

The 555M run uses a 100,000-token BPE vocabulary learned from a sample of web text, code and French. The trainer in [`lang/bpe.c`](../lang/bpe.c) reads the sample in chunks, keeps only distinct words with their counts, and picks each merge from a heap. Runs of spaces are their own words, so code indentation becomes a few tokens instead of a pile of single spaces.

[Subword BPE tokenizer](subword-bpe-tokenizer.md) explains BPE itself. This note is about making it hold at 100k merges and gigabytes of text.

## What problem it solves (plain language)

The textbook loop rereads the whole text for every merge. At 100k merges on a 1.5 GB sample that never finishes. And the 16k vocabulary was learned on web text: it spells code one or two characters at a time, and eight spaces of indentation cost eight tokens.

## How it works for a human (intuition, analogy)

Do not recount the book for every decision. Write each different word once on a card with how many times it appears, and work on the cards. Keep the pair counts in a ledger, and after a merge only correct the lines that the merge touched.

## Formal view (definitions, formulas, shapes, algorithms)

**Collect.** The text is cut into words: an optional leading space, then a run of one class (letters, digits and UTF-8 bytes, or everything else), at most 64 bytes. Each distinct word is stored once with its count `c(w)` in a hash table.

**Count.** For every adjacent pair `(a, b)` in word `w`: `count(a, b) += c(w)`.

**Merge loop**, `vocab_size - 260` times (4 specials and 256 bytes come first):

1. Pop the heap until the top entry's count is still the real one. A stale entry is pushed back with its real count.
2. Record the merge `(a, b) -> new id`.
3. In every word that still has two symbols or more and contains `(a, b)`: replace it, subtract `c(w)` from the pairs that disappeared, add it to the pairs that appeared.
4. Push the pairs whose count changed.

Ties go to the smaller `(left, right)`, so the same sample always gives the same vocabulary.

**Encode.** Per word: find the pair with the lowest merge rank (hash lookup on `(left, right)`), merge it, repeat.

## Complexity / numerical pitfalls / design choices

- **Memory follows distinct words, not bytes.** A file of any size is read 1 MiB at a time, cut at the last space or newline so no word is split.
- **Lazy heap.** Counts change all the time; fixing the heap at each change would cost more than leaving stale entries and checking them when they surface.
- **Words drop out.** A word merged down to one symbol leaves the active list for good.
- **Space runs (`ws_runs`).** With plain splitting, each space of an indentation starts its own word. Now the spaces before a space form one word, and the last space still leads the word that follows, so `␣␣␣␣return` is `␣␣␣` + `␣return`. Merges can then learn one token for a whole indentation.
- **Old vocabularies keep their splitting.** A vocab file says `BPEV 2` when it uses space runs and `BPEV 1` otherwise; a checkpoint carries the same flag in its merge count. The 16k models encode exactly as before.
- **Sample, do not read everything.** The vocabulary is learned from ten evenly spaced slices of each source; the full text is only encoded.

## How it shows up in this repo

- [`lang/bpe.h`](../lang/bpe.h): `bpe_trainer_create`, `bpe_trainer_add` (call it as often as needed), `bpe_trainer_finish(t, vocab_size)`, `bpe_trainer_free`; `bpe_train_from_file(path, vocab_size, max_bytes)` wraps them.
- `bpe_encode_n(b, text, len, ids, max_ids)` encodes a buffer by length, for pieces that are not NUL-terminated.
- [`scripts/bpe_train.c`](../scripts/bpe_train.c): `bpe_train TEXT VOCAB_SIZE OUT [max_bytes]`.
- [`scripts/build_code.sh`](../scripts/build_code.sh): builds the sample, trains `data/mix100k.vocab`, then encodes every source in parallel (the text is split at line ends, one `scripts/pretok_ids` per piece, ids joined in order).
- Tests in [`tests/test_ann.c`](../tests/test_ann.c): round trip, determinism, and the space-run splitting.

## Use

```text
JOBS=4 SAMPLE_MB=1500 CODE_PCT=35 FR_PCT=15 VOCAB=100000 scripts/build_code.sh
```

## What next

- Step 3 still looks at every active word for each merge. An index from pair to words would make late merges, which touch few words, nearly free.
