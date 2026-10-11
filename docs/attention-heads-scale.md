# Attention heads scale (R5b)

## Abstract

`LM_N_HEADS=16` at D=512 keeps `d_head=32`. Heads must divide D.

## How it shows up in this repo

- [`lang/lm_config.h`](../lang/lm_config.h)
