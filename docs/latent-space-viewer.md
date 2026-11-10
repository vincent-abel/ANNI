# Latent space viewer

## Abstract

`anni-probe` reads a checkpoint and writes what the model holds inside: its **token table** and the **hidden states** of real text at every layer. `anni-latent` is a native window (C99, GLFW, OpenGL 3.3, cimgui) that shows those vectors as a 3D cloud, with a layer slider, a search box, nearest neighbours, and the comparison of two checkpoints. It reads files only: the training code is not touched and no GPU work is needed beyond drawing points.

## What problem it solves (plain language)

The loss says how well the model predicts, not what it has organized. Looking at the vectors shows it directly: which tokens the model treats as close, how a word moves through the layers, whether French, English and code live apart, and where the points it predicts badly sit.

## How it works for a human (intuition, analogy)

A vector of 1536 numbers cannot be seen, so the viewer draws a **map**. Like a map of the Earth, it distorts: two points close on the map are usually close in the real space, but distances between far regions mean little. For that reason neighbours are never read from the map. A click measures them again on the full 1536 numbers and draws them as lines.

## Formal view (definitions, formulas, shapes, algorithms)

**Dump.** `anni-probe --model F --out DIR [--data FILE:LABEL]... [--positions N]` writes:

| File | Content |
| --- | --- |
| `vocab.npy` | token table, `[V, D]` |
| `vocab.txt` | one token per line |
| `states.npy` | hidden states, `[L + 1, N, D]`, float16 |
| `points.tsv` | per point: source, window, position, token, next token, predicted token, probability, NLL |

With `L` blocks there are `L + 1` layers: the input of each block, read from the training cache after `lm_forward_cache`, then the last output. The largest absolute value seen on the 555M model is about 317, well inside float16.

**Layout.** Vectors are centred, the `k` nearest neighbours are found by cosine in the **full space**, then UMAP places them in 3D. Reducing with PCA first is not usable here: on the 100k token table, 50 dimensions keep 27% of the true 15 neighbours, 128 keep 46%, 256 keep 59%, 512 keep 71%.

**Alignment.** A layout alone has an arbitrary orientation, so a slider over independent layouts would only show noise. Each layer starts from the layout of the layer before (80 epochs, 300 for the first), and a compared dump starts from the layout of the first one. Points then move only when their neighbourhood changes. The viewer interpolates between layers and between checkpoints.

**Cache.** Layouts are computed at the first opening and stored in the dump as `layout-vocab.f32` and `layout-text.f32`. Later openings are immediate.

## Use

```text
make latent
./anni-probe --model models/run555m-base-0.653g.bin --out models/latent/0.653g \
    --data data/mix100k_val_web_ids.bin:web --data data/mix100k_val_fr_ids.bin:fr \
    --data data/mix100k_val_code_ids.bin:code
./anni-latent models/latent/0.653g
./anni-latent models/latent/0.653g --compare models/latent/1.2g --scene text
```

`make latent` fetches cimgui through `scripts/latent_deps.sh` and needs `libglfw3-dev`.

Left drag orbits, right drag looks, the wheel zooms, W A S D Q E fly. A click selects a point and draws its neighbours, a double click flies to it. `--shot FILE.ppm` renders one frame and quits, which also works without a screen under `xvfb-run` with software GL.

## Limits

- The map is a projection: only the neighbour lines are exact.
- Text points are a sample of the validation files, not the corpus.
- The first opening of a dump is slow: the full-space graph of the token table is about 1.5e13 multiply-adds on the CPU, 10 minutes on 3 threads next to a training run. The 15 text layers take about as long.
- Layouts are scaled by the median radius, so a few strays may sit far outside the cloud.
