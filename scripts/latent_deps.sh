#!/bin/sh
# Fetch Dear ImGui and its C binding, at a fixed commit, for `make anni-latent`.
set -e
COMMIT=2a6a93fccd10f275ee9206167dc9698f6f00fac1
DIR=latent/third_party/cimgui
[ -f "$DIR/imgui/imgui.h" ] && exit 0
rm -rf "$DIR"
git clone -q https://github.com/cimgui/cimgui "$DIR"
git -C "$DIR" checkout -q "$COMMIT"
git -C "$DIR" submodule -q update --init --depth 1
