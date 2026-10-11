# Repro JSON sidecar (E11)

## Abstract

Training jobs write a JSON flag dump next to the checkpoint so runs are reproducible.

## How it shows up in this repo

- `./autotrain` writes `OUT.autotrain.json` (AT5 / E11)
- Manual train can grow the same pattern later
