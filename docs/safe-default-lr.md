# Safe default learning rate (E1)

## Abstract

The teaching default `--lr 0.01` blows up FP32 runs at D≥256. E1 changes the CLI (and tower) default to **0.001**.

## What problem it solves

Unstable training wastes GPU hours and produces NaN checkpoints. A safer default makes first runs useful without a schedule.

## How it works for a human

Omit `--lr` and you get 0.001. Pass `--lr` explicitly when sweeping or using Autotrain.

## Formal view

Adam step \(\theta\leftarrow\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)\). At FP32 with larger \(D\), \(\eta=10^{-2}\) often drives logits into overflow before the moments adapt; \(\eta=10^{-3}\) keeps updates in a typical LM range until warmup/cosine (E2) lands.

## Complexity / pitfalls / design choices

- Phrase demos / legacy `lm_train_steps` may still use 0.01 locally.
- Schedule peak in E2 can temporarily exceed 0.001 with warmup.

## How it shows up in this repo

- [`main.c`](../main.c): default `--lr` 0.001
- [`cli/tower.c`](../cli/tower.c): tower fallback 0.001

## Tiny worked example

D=256, lr=0.01 → NLL → NaN within an epoch on toy text. Same run at 0.001 stays finite.

## What comes next

[LR schedule, clip, weight decay](lr-schedule-clip-wd.md) (E2).
