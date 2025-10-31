#!/usr/bin/env bash
set -e

mkdir -p results
mkdir -p plots/quant_mlp
mkdir -p plots/quant_resnet20



: '
# run quantization/MI sweep for MLP (MNIST)
python quant_mi_analysis.py \
  --model_name MLP \
  --output_path results/quant_analysis_mlp.json

# make plots for MLP run
python plot_quant.py \
  --input_json results/quant_analysis_mlp.json \
  --out_dir plots/quant_mlp
'


# run quantization/MI sweep for ResNet20 (CIFAR)
python quant_mi_analysis.py \
  --model_name ResNet20 \
  --output_path results/quant_analysis_resnet20.json


# make plots for ResNet20 run
python plot_quant.py \
  --input_json results/quant_analysis_resnet20.json \
  --out_dir plots/quant_resnet20
