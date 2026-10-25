# Talking model chat fine-tune (talk-170m)

## Abstract

After pretraining, the model is fine-tuned for two epochs on 24.2M tokens of `User:` / `Assistant:` conversations ([data note](talk-170m-data.md)). The loss counts only what the assistant says: the user's words and the `Assistant:` cue that follows them are masked out (35% of the tokens). A masked position gets no loss and no gradient, on the host path and on the GPU kernel alike.

## What problem it solves

Fine-tuning on whole transcripts also trains the model to write the user's side. That wastes a third of the signal on text the model never has to produce, and it teaches the model to keep going past its own answer into an invented next question. Masking turns the same data into "given this conversation, answer". `chat_reply_cut` then stops a reply at the next `User:` line, which the model learns to emit because that marker is not masked.

## Formal view

For a sequence \(x_1..x_T\) with mask \(m_t \in \{0,1\}\) on the token being predicted:

\[
\mathcal{L} = \frac{1}{\sum_t m_{t+1}} \sum_{t=1}^{T-1} m_{t+1}\,\bigl(-\log p_\theta(x_{t+1}\mid x_{\le t})\bigr),
\qquad
\frac{\partial \mathcal{L}}{\partial z_t} = m_{t+1}\,\bigl(\mathrm{softmax}(z_t) - e_{x_{t+1}}\bigr).
\]

The model still reads masked tokens as input; only their prediction is skipped.

## How it shows up in this repo

- [`scripts/pretok_chat.c`](../scripts/pretok_chat.c): writes `*_mids.bin`, where a masked token is stored as \(-(\mathrm{id}+1)\). Each turn is encoded in segments that start at a space or a line break, where BPE words split anyway.
- [`lang/dataset_text.c`](../lang/dataset_text.c): a `*_mids.bin` file loads with `masked = 1`. `text_dataset_window` always returns decoded ids, so evaluation and the embedding never see a negative id; `text_dataset_window_mask` also returns the mask.
- [`lang/lm.c`](../lang/lm.c): `lm_loss_set_mask` turns masked targets into \(-1\). `ce_rows_backward` zeroes those rows and averages over target rows only.
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `ce_onehot_ce_kernel` zeroes the softmax row of a \(-1\) target.
- [`lang/lm_train.c`](../lang/lm_train.c): the corpus loop fills the mask per window and sets it around the packed loss. CUDA graphs are skipped while a mask is present, because the mask is read on the host when targets are built.
- Tests: `test_text_dataset_masked_ids`, `test_lm_loss_mask`. With an all-ones mask the loss equals the unmasked loss; with a sparse mask only the kept positions get a hidden gradient.

## Run

`scripts/talk170m_run.sh` runs these steps after pretraining: a clean CUDA rebuild with this code, `scripts/pretok_chat` on the train and val text, then

```bash
./anni lm train --resume models/talk170m.bin --out models/talk170m-chat.bin --force \
  --data data/chat_train_mids.bin --val-data data/chat_val_mids.bin \
  --tokenizer bpe --vocab data/mix.vocab \
  --window 256 --stride 256 --batch 16 --grad-accum 8 --epochs 2 \
  --lr 5e-5 --lr-decay none --grad-clip 1.0 --weight-decay 0.1 \
  --flash-attn --bf16 --eval-every 1 --eval-windows 256 --seed 2 -d
```

That is 94k windows and about 735 steps per epoch, roughly 25 minutes for both epochs at pretraining speed. The learning rate is constant: the schedule follows Adam's step count, which `--resume` carries over from pretraining and which is already past the cosine decay. The driver then restarts `janni-llama-server`.

Talk to the result with `./anni chat --model models/talk170m-chat.bin`.

## Pitfalls

- The GPU half of the masking (the kernel row zeroing and the device-mean rescale) compiles but cannot run while pretraining holds the GPU. Its first real use is the fine-tune itself: check that the logged NLL is in the same range as the held-out NLL, not near 0 or above the pretraining loss.
- Held-out loss on `chat_val_mids.bin` counts every token, user turns included, so it reads higher than the masked training loss.
- Conversations are concatenated and cut into 256-token windows, so a window can start mid-conversation; the mask still drops the user's side.
