#!/usr/bin/env bash

mkdir -p plots/quant_mlp
mkdir -p plots/quant_resnet20


# make plots for MLP run
python plot_quant.py \
  --input_json results/quant_analysis_mlp.json \
  --out_dir plots/quant_mlp


# make plots for ResNet20 run
python plot_quant.py \
  --input_json results/quant_analysis_resnet20.json \
  --out_dir plots/quant_resnet20


# make plots for MLP run
python plot_pruning.py \
  --input_json results/pruning_all_layers_mlp.json \
  --out_dir plots/pruning_mlp

# make plots for ResNet20 run
python plot_pruning.py \
  --input_json results/pruning_all_layers_resnet20.json \
  --out_dir plots/pruning_resnet20


python plot_factorization.py \
  --input_json results/factorization_analysis_resnet20.json \
  --out_dir plots/factorization_resnet20
