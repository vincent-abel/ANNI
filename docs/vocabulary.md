# Vocabulary

## Abstract

A vocabulary is the closed list of symbols the model knows, each with a stable integer id. This module builds a byte vocabulary plus four special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`).

## What problem it solves

Ids alone are opaque. Training logs, generation, and tooling need names for tokens. Specials mark padding, sequence start/end, and unknowns. The vocab is the single source of truth for “how many logits does the LM head emit?”

## How it works for a human

Think of a dictionary where entry #0 is `<pad>`, #1 `<bos>`, #2 `<eos>`, #3 `<unk>`, then #4 is byte 0, #5 byte 1, … up through all 256 bytes. Looking up a string walks the table; looking up an id indexes it directly.

## Formal view

Let \(V = S \cup B\) where \(|S|=4\) specials and \(|B|=256\) bytes. Total size \(N = 260\).

\[
\operatorname{id}(t) = \begin{cases}
i & t = \text{name of entry } i \\
\operatorname{id}(\texttt{<unk>}) & \text{otherwise}
\end{cases}
\]

Byte \(b\) lives at id \(4 + b\). The tokenizer encodes text into these shifted ids.

Complexity: `vocab_token` is \(O(1)\); `vocab_id` is \(O(N)\) linear scan (fine at \(N=260\); a hash map can replace it later).

## Complexity / pitfalls / design choices

- Printable ASCII bytes store as single-character strings; others as `<bN>` labels for debugging.
- Linear `vocab_id` is acceptable for toy scale only.
- Freeing must release every duplicated string and the pointer table.

## How it shows up in this repo

- [`lang/vocab.h`](../lang/vocab.h) / [`lang/vocab.c`](../lang/vocab.c): `vocab_byte_create`, `vocab_free`, `vocab_size`, `vocab_token`, `vocab_id`
- Constants `VOCAB_PAD` … `VOCAB_UNK` and `VOCAB_SPECIALS`
- Tokenizer encode/decode use the same id space

## Tiny worked example

`'A'` (65) → id `4+65=69` → `vocab_token(69)` → `"A"`.

## What comes next

Token embeddings allocate a \(N \times d_{\text{model}}\) matrix indexed by these ids.
