#!/usr/bin/env bash
# Build a copy of HEAD with the 555M sizes in build/hellaswag, for the HellaSwag
# bench of scripts/run555m_checks.py. Apart from build/555m, whose binary trains.
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD=build/hellaswag
CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-13* | tail -1)}

rm -rf "$BUILD"
mkdir -p "$BUILD"
git archive HEAD | tar -x -C "$BUILD"
sed -i -e 's/\(define LM_D_MODEL\t\).*/\11536/' \
    -e 's/\(define LM_N_HEADS\t\).*/\124/' \
    -e 's/\(define LM_D_FF\t\).*/\16272/' \
    -e 's/\(define LM_N_BLOCKS\t\).*/\114/' "$BUILD/lang/lm_config.h"
nice -n 10 make -C "$BUILD" -j3 cuda CUDA_HOME="$CUDA_HOME"
