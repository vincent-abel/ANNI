# CUDA streams overlap (R8 / E6)

## Abstract

A non-blocking copy stream stages id H2D for embed gather; compute stays on the default stream. Events can replace blanket sync later.

## How it shows up in this repo

- `gpu_copy_stream`, async H2D in `gpu_embed_lookup`
