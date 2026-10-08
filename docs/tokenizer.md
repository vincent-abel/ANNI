# Tokenizer

## Abstract

A tokenizer turns human text into a sequence of integer token ids the model can consume, and turns ids back into text. This repo starts with a **byte/character tokenizer**: each byte is its own token id in `0..255`.

## What problem it solves

Neural nets do not read letters. They read numbers. Someone must map `"Hi"` → `[72, 105]` and later map predictions back to characters. Without a stable encode/decode pair, training and generation cannot share the same meaning of “token 65.”

## How it works for a human

Imagine labeling every byte in a file with its numeric value (ASCII/UTF-8 bytes). Encoding walks the string once and writes those labels into an array. Decoding walks the ids and writes characters (unknown ids become `?`). There is no merge step and no word list yet—that comes with a richer vocabulary later.

## Formal view

Let text be a byte string \(s = s_0 s_1 \ldots s_{n-1}\) with \(s_i \in \{0,\ldots,255\}\).

With specials reserved at the front of the vocab (\(S=4\)):

\[
\operatorname{enc}(s)_i = S + s_i, \qquad
\operatorname{dec}(z)_i =
\begin{cases}
\operatorname{chr}(z_i - S) & 0 \le z_i - S < 256 \\
\texttt{?} & \text{otherwise}
\end{cases}
\]

API:

- `tokenize_encode(text, ids, max_ids) → n` writes up to `max_ids` ids, returns count or `-1`.
- `tokenize_decode(ids, n_ids, out, out_cap) → n` NUL-terminates `out`.

Complexity: \(O(n)\) time and \(O(1)\) extra space beyond the output buffer.

## Complexity / pitfalls / design choices

- **Vocabulary size** is fixed at 256 (`LM_BYTE_VOCAB`), which is tiny versus BPE/WordPiece (10k–50k) but perfect for debugging a toy LM.
- **Unicode:** multi-byte UTF-8 characters become several tokens (one per byte). That is intentional for v1.
- **Truncation:** encode silently caps at `max_ids`; callers must check the returned length against `strlen`.
- **No special tokens** yet (`<bos>`, `<eos>`); those arrive with the Vocabulary module and will wrap this encoder.

## How it shows up in this repo

- [`lang/tokenize.h`](../lang/tokenize.h) / [`lang/tokenize.c`](../lang/tokenize.c)
- Shared sizes in [`lang/lm_config.h`](../lang/lm_config.h)
- Built via `LANG_SRC` and `-Ilang` in the Makefile

## Tiny worked example

`"AB"` → ids `[65, 66]` → decode → `"AB"`.

## What comes next

Vocabulary adds named specials and id↔string maps on top of (or beside) this byte encoding.
