#!/usr/bin/env bash
set -e

mkdir -p results
mkdir -p plots/pruning_mlp
mkdir -p plots/pruning_resnet20

: '
# run pruning / MI sweep for MLP (MNIST)
python prune_mi_analysis.py \
  --model_name MLP \
  --output_path results/pruning_all_layers_mlp.json

# make plots for MLP run
python plot_pruning.py \
  --input_json results/pruning_all_layers_mlp.json \
  --out_dir plots/pruning_mlp
'


# run pruning / MI sweep for ResNet20 (CIFAR)
python prune_mi_analysis.py \
  --model_name ResNet20 \
  --output_path results/pruning_all_layers_resnet20.json

# make plots for ResNet20 run
python plot_pruning.py \
  --input_json results/pruning_all_layers_resnet20.json \
  --out_dir plots/pruning_resnet20
