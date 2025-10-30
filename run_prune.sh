#!/usr/bin/env bash
set -e

mkdir -p results
mkdir -p plots

python prune_mi_analysis.py

python plot_pruning.py \
  --input_json results/pruning_all_layers.json \
  --out_dir plots/pruning_all_layers

