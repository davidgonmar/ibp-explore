#!/usr/bin/env bash
set -e

mkdir -p results
mkdir -p plots

python quant_mi_analysis.py

python plot_quant.py \
  --input_json results/quant_analysis.json \
  --out_dir plots/quant_analysis
