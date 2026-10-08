# Generation from checkpoints

## Abstract

After [`lm_save`](lm-weight-serialization.md) writes a trained checkpoint, **inference** should not require re-running the training CLI. **`lm_load`** restores weights, then **[`lm_generate`](../lang/generate.c)** runs autoregressive decoding: repeated **`lm_forward`** on the growing prefix, **LM head logits at the last position**, sample or argmax next byte, append until `max_tokens` or EOS.

## What problem it solves (plain language)

Checkpoints are useful only if you can **type a prompt and read continuations** that reflect learned weights. Generation proves the saved file round-trips correctly and that training improved predictions beyond random init.

## How it works for a human (intuition, analogy)

Load the saved “brain,” show it the beginning of a sentence, ask repeatedly: “what byte next?” Append each answer and ask again—like autocomplete with no backspace ([text-generation.md](text-generation.md)).

## Formal view (definitions, formulas, shapes, algorithms)

**Prompt** token ids length \(T_0 \ge 1\).

**Step \(k\)** (generate up to `max_gen` bytes):

1. Let current length \(n = T_0 + k\) (cap \(n \le\) `LM_MAX_CTX`).
2. `lm_forward(H, model, ids, n)`.
3. `lm_logits_at(logits, model, H, n-1)` — **last** position only.
4. Next id: greedy `argmax` if `temperature <= 0`, else sample from `softmax(logits / τ)`.
5. Append id; stop if id is EOS/pad policy or buffer full.

**Load path.**

```text
model = lm_load(checkpoint_path)
if (!model) error
lm_generate(model, prompt, out_buf, out_cap, max_tokens, tau)
lm_free(&model)
```

**Planned CLI.**

```text
./ann lm generate --checkpoint checkpoints/toy.bin --prompt "hello" --max 32 --temp 0.8
```

**Consistency checks after load.**

- `vocab_size` matches embed rows.
- `lm_nll` on a fixed phrase matches value logged at save time (sanity).

## Complexity / numerical pitfalls / design choices (CS depth)

**Context limit.** Generation cannot exceed `LM_MAX_CTX`; truncate **left** prefix (keep tail) if implementing long prompts later—not in v1.

**Exposure bias.** Training uses teacher forcing; generation feeds model outputs—short toy continuations may loop or drift.

**Temperature.** τ→0 → greedy; large τ → uniform noise; match [`generate.c`](../lang/generate.c) behavior.

**Double precision.** Same as training; no batching required for toy CLI.

**Security.** `lm_load` untrusted paths—same caveats as serialization doc.

## How it shows up in this repo

- Today: [`lang/generate.c`](../lang/generate.c) `lm_generate`; [`main.c`](../main.c) trains in-memory then generates—**no load yet**.
- Planned: `lm_load` + CLI subcommand `generate` ([lm-cli.md](lm-cli.md)).
- Depends on: [lm-weight-serialization.md](lm-weight-serialization.md).

## Worked example

```text
./ann lm train --corpus data/toy.txt --out checkpoints/toy.bin --epochs 5
./ann lm generate --checkpoint checkpoints/toy.bin --prompt "The " --max 16 --temp 0
```

Expect continuation bytes influenced by toy corpus statistics (not random ASCII). If load broken, output matches **fresh `lm_create`** gibberish—regression signal.

## What next

- Wire `lm_load` into `run_lm` / dedicated generate handler.
- Optional: print per-step logits top-3 for debugging.
- Test: save/load then `strcmp` generated greedy output before/after round-trip.
