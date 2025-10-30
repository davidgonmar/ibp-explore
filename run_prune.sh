#!/usr/bin/env bash
set -e

mkdir -p results
mkdir -p plots

python prune_mi_analysis.py
python prune_mi_analysis_no_last_layer.py
python prune_analysis_with_probe.py

python plot_pruning.py \
  --input_json results/pruning_all_layers.json \
  --out_dir plots/pruning_all_layers

python plot_pruning.py \
  --input_json results/pruning_no_last_layer.json \
  --out_dir plots/pruning_no_last_layer

python plot_pruning.py \
  --input_json results/pruning_with_probe.json \
  --out_dir plots/pruning_with_probe
