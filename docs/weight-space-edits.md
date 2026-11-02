# Weight-space edits on a 185M model

## Abstract

We tried to make the chat model talk170m-2k-chat better at coding by editing its weights, with no training: no optimizer and no gradients, a few forward passes per edit. The method follows Weight-Space Programming [1]. It projects one direction out of the FFN output weights of one block. We ran it twice, 25 rounds each.

- **Edits aimed at text loss do nothing useful.** On held-out code and web text, every one of the 25 edits made the NLL worse, by +0.0001 at best.
- **Edits aimed at a reply habit work.** Here the direction separates the model's usual reply ("Here's how…" and a ```python block) from a reply in the format the coding bench applies (`File: <path>` and a code block). Removing that direction from block 9 lowered the NLL of the bench format by 0.12 nats per token on held-out tasks. It closed 30% of the gap to the habitual reply, and NLL on code and web text did not get worse.

So on a small model, the article's main finding holds. An edit can remove an obstacle but cannot add a skill. The NLL on the model's own training distribution has no obstacle to remove, because training already minimised it.

## Background: Weight-Space Programming

The article [1] has a language model read another network's weights and write new values directly. Its language-model result, Experiment 8, is on Qwen3.5-0.8B:

- A "looping" direction, written by the MLPs of layers 15 and 19, is projected out of 64 output columns per layer. That is 131k weights, 0.016% of the model.
- MATH-500 goes from 7.9% to 48.1%, and GSM8K from 13.8% to 67.1%.
- Generations that hit the token limit drop from 93% to 33%.
- The direction comes from training-set practice problems, never from the test sets.

The article also reports the limit: on Gemma 3 270M, edits left MATH-500 at about 7%. The ability has to be in the model already.

Our question: does any of this carry over to a 185M-parameter model trained here from scratch, on a 5070 Ti, with a 25-round budget?

## Method

The FFN of block \(l\) adds \(o = \mathrm{FFN}(\mathrm{ln}_2(x_{mid}))\) to the residual stream, where \(o = h W_2 + b_2\), \(W_2\) is \(D_{FF} \times D\), and \(h\) are the hidden activations.

**Statistics.** For two classes of positions, A and B, [`scripts/lm_edit.c`](../scripts/lm_edit.c) records per block:

- the mean of \(o\), \(m_A\) and \(m_B\);
- the second moment of \(o\), \(C_A\) and \(C_B\).

**Directions.** From these come three unit directions per block:

- **mean:** \(m_A - m_B\), normalised;
- **B-specific:** the top eigenvector of \(C_B - C_A\), what the block writes more in class B;
- **A-specific:** the top eigenvector of \(C_A - C_B\).

**The edit** `L:DIR:K:BETA` takes the \(K\) rows \(w_i\) of block \(L\)'s \(W_2\) with the largest \(|w_i \cdot u|\) (\(K=0\) means every row, plus \(b_2\)). It sets

\[ w_i \leftarrow w_i - \beta\,(w_i \cdot u)\,u \]

- \(\beta = 1\) projects \(u\) out of what those rows can write.
- \(\beta > 1\) pushes the output to the opposite side.
- \(\beta < 0\) amplifies \(u\).

Edits can be stacked with `+`.

**Rounds.** Each round applies one edit, uploads \(W_2\) to the GPU, scores the model, and undoes the edit. Once the directions are cached, a round takes about 1 s: forward passes of 2 × 2048 tokens in BF16 with flash attention.

## Experiment 1: code vs web text

- **Model:** talk170m-2k, the pretrained model extended to 2048 tokens.
- **Classes:** A is 24 code windows and B is 24 web windows, all from the training ids.
- **Score:** mean NLL on 24 held-out code windows and 24 held-out web windows, 2048 tokens each.
- **Unedited:** code 1.4203, web 3.2467.

| Round | Edit | Code Δ | Web Δ |
|---|---|---|---|
| 18 | 9:web:64:0.5 | +0.0001 | +0.0003 |
| 9 | 9:web:64:1 | +0.0003 | +0.0011 |
| 23 | 8:code:0:−0.25 | +0.0012 | +0.0004 |
| 7 | 6:code:0:−0.5 | +0.0041 | +0.0008 |
| 3 | 9:web:0:1 | +0.0057 | +0.0229 |
| 4 | 11:web:0:1 | +0.0748 | +0.3644 |
| 11 | 1:web:0:1 | +3.6713 | +6.4995 |

All 25 rounds are in `logs/edit/batch{1,2,3}.log`.

- **None of the 25 edits lowered either NLL.**
- **Stacking doesn't help.** Stacking the three mildest edits (round 25) added up their harm: +0.0060 code.
- **Blocks 1 and 2 are fragile.** One direction removed at block 1 wrecks the model.
- **The middle blocks barely react.**

Before this experiment, a first try added the code-minus-web mean residual to one FFN bias \(b_2\), instead of projecting a direction out. It also made both NLLs worse, by +0.12 at a scale of 0.25 on block 1. Shifting the output by a mean vector is not the article's method, and it did not work here either.

## Experiment 2: edit-format replies vs talk replies

- **Model:** talk170m-2k-chat. On the 20 single-turn coding tasks of [`taskbench/`](../taskbench/README.md) it passes 0 of 20 and never changes a file. Its replies start "Here's how you can implement…", followed by Python for a JavaScript project.
- **Data:** [`scripts/edit_pairs.js`](../scripts/edit_pairs.js) builds, for each task:
  - the bench prompt;
  - an **edit** reply: `File: <first shown file>` and the start of that file in a ``` block (for the read-only question, the right answer);
  - a **talk** reply: the model's own greedy reply from the bench run.
- **Positions:** the ones that predict the first 64 reply tokens.
- **Split:** the directions come from the 10 even-numbered tasks. The score is on the 10 odd-numbered tasks.
- **Score:**
  - the NLL of the edit reply;
  - the margin, talk NLL − edit NLL, which is negative while the model prefers to talk;
  - the code and web windows again, as a damage check.
- **Unedited:** edit NLL 1.3730, talk NLL 0.5238, margin −0.8492.

| Round | Edit | Edit NLL Δ | Margin Δ | Code Δ | Web Δ |
|---|---|---|---|---|---|
| 4 | 9:talk:0:1 | −0.1179 | +0.1887 | −0.0055 | −0.0016 |
| 14 | 9:talk:1024:1 | −0.1088 | +0.1630 | −0.0066 | −0.0034 |
| 22 | 9:talk:0:1.25 | −0.1244 | +0.2381 | −0.0016 | +0.0003 |
| **21** | **9:talk:1024:1.5** | **−0.1229** | **+0.2496** | **−0.0006** | **−0.0016** |
| 11 | 9:talk:0:1.5 | −0.1263 | +0.2905 | +0.0041 | +0.0029 |
| 24 | 9:talk:0:1.5 + 8:talk:0:1 | −0.1242 | +0.3009 | +0.0111 | +0.0081 |
| 12 | 9:talk:0:2 | −0.1195 | +0.4089 | +0.0206 | +0.0104 |
| 3 | 8:talk:0:1 | −0.0170 | +0.0218 | +0.0029 | +0.0036 |
| 1 | 3:talk:0:1 | +0.0441 | −0.0399 | +0.0186 | +0.0036 |

All 25 rounds are in `logs/edit/pairs_batch{1,2,3}.log`.

**One block carries the habit.** Removing the "talk" direction from block 9 does nearly all the work, and the even (direction) tasks and the odd (held-out) tasks move together. Other blocks give about a tenth of the effect or less, and stacking them adds damage faster than gain. Like the article's 64 columns, the edit touches a small part of one matrix.

**Strength trades margin for damage.** Edits beyond \(\beta = 1.5\) on every row keep raising the margin but start to cost code NLL. Limiting the edit to the 1024 most aligned rows (a quarter of \(W_2\)) keeps the gain and removes the cost.

**The kept edit.** Round 21 is saved as `models/talk170m-2k-chat-edit.bin`, a copy; the original is untouched. Reloaded and scored again, it gives the same numbers: edit NLL 1.2501, margin −0.5996, code 1.7323 (was 1.7329), web 3.7343 (was 3.7359).

## Discussion

**What carried over from [1]:**

- **Remove, don't add.** Projecting a direction out worked; shifting by a mean vector, or amplifying the "code" or "edit" directions, did not.
- **The direction must describe a behaviour, not a domain.** "Code vs web" is knowledge the model needs, so removing any part of it hurts. "Talk reply vs edit reply" is a habit from the chat fine-tune, and removing it costs nothing.
- **It's cheap.** About 25 forward-pass rounds, under a minute of GPU time, edited at one block.

**What we did not show:**

- **No bench result.** NLL of a format is a proxy. We have not rerun the bench, so we don't know whether greedy generations now start with `File:`.
- **It can't pass the tasks.** Even if the replies now use the format, the model still writes wrong code, often in the wrong language. This is the article's Gemma 270M result: a small model lacks the ability, and no edit adds it.
- **Small samples.** The direction comes from 10 tasks of about 64 tokens each, and the score from the other 10. The test prompts come from the same bench, so a new set of tasks would be a stronger check.

## Reproduce

```
make cuda          # then build scripts/lm_edit as its header says
node scripts/edit_pairs.js logs/taskbench/talk170m-2k-chat.json > logs/edit/pairs.txt
LM_EDIT_B=2 LM_EDIT_PAIRS=logs/edit/pairs.txt LM_EDIT_DIRS=logs/edit/dirs_pairs.bin \
    scripts/lm_edit models/talk170m-2k-chat.bin 9:talk:0:1 9:talk:1024:1.5
LM_EDIT_B=2 LM_EDIT_PAIRS=logs/edit/pairs.txt LM_EDIT_DIRS=logs/edit/dirs_pairs.bin \
    LM_EDIT_SAVE=models/talk170m-2k-chat-edit.bin \
    scripts/lm_edit models/talk170m-2k-chat.bin 9:talk:1024:1.5
```

Without `LM_EDIT_PAIRS`, the tool runs Experiment 1 instead, on code and web windows. Use `LM_EDIT_B=2` at 2048 tokens: two windows per forward fit in 16 GB.

## References

1. RikiyaT, *Weight-Space Programming*. <https://rikiyat.github.io/blog/weight-space-programming/>. Run logs: <https://github.com/RikiyaT/lm_understand-logs>.

## What comes next

- Rerun [`taskbench/`](../taskbench/README.md) on `models/talk170m-2k-chat-edit.bin` to see whether greedy replies now use the `File:` format.
- Look for a second obstacle in the bench replies: looping, and Python in a JavaScript project.
- See [Talking model chat fine-tune](talk-170m-chat.md) for where the talk habit comes from.
