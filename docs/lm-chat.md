# Interactive chat (AnnI)

## Abstract

`./anni chat` is Stage-1 multi-turn chat over a pretrained checkpoint. On a TTY with ncurses it opens a messenger-style UI; otherwise it uses a stdin line loop. Each turn appends plain-text `User:` / `Assistant:` prefixes, calls `lm_generate`, and trims history so the prompt stays within `LM_MAX_CTX`.

## What problem it solves

After Target quality (Q4), you want local chat without Stage-3 SFT templates or an HTTP server.

## How it works for a human

```text
./anni chat --model wiki_bpe.bin
# TTY: full-screen AnnI chat (Enter send, PgUp/PgDn scroll, q quit)
# pipe / no ncurses:
User: What is a neuron?
Assistant: …
(empty line or Ctrl-D to exit)
```

Flags: `--model`, `--tokens`, `--temperature`, `--top-k`, `--top-p`, `--vocab`.  
Alias: `./anni lm chat …` (same handler). Compat: `./ann` → `./anni`.

## Formal view

Maintain history string \(H\). On user utterance \(u\):

\[
H \leftarrow H + \texttt{User: }u\texttt{\textbackslash nAssistant:}
\]

Trim a prefix of \(H\) until \(\mathrm{encode}(H) \le L_{\max} - N_{\mathrm{new}}\), then sample continuation and append the reply to \(H\).

## Complexity / pitfalls / design choices

- **Plain prefixes only:** no special role tokens (Stage-3 TOK2/TOK3).
- **Quality gated by pretrain:** garbage generate ⇒ chat will not help.
- **Context trim:** drops oldest lines.

## How it shows up in this repo

- [`cli/chat_session.c`](../cli/chat_session.c) — history + generate
- [`cli/chat_ui.c`](../cli/chat_ui.c) — ncurses UI
- [`main.c`](../main.c) — `./anni chat` dispatch
- Roadmap: **CHAT1** / **CHAT2** / **CHAT3** in [`../README.md`](../README.md)

## Tiny worked example

```text
$ ./anni chat --model wiki_bpe.bin --tokens 64 --temperature 0.8 --top-p 0.9
```
