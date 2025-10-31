#!/usr/bin/env bash
set -e
mkdir -p results
mkdir -p plots/factorization_mlp
mkdir -p plots/factorization_resnet20


: '
python factorization_mi_analysis.py \
  --model_name MLP \
  --output_path results/factorization_analysis_mlp.json
run python plot_factorization.py \
  --input_json results/factorization_analysis_mlp.json \
  --out_dir plots/factorization_mlp
'



python factorization_mi_analysis.py \
  --model resnet20 \
  --dataset cifar10 \
  --checkpoint resnet20_cifar.pth \
  --out_json results/factorization_analysis_resnet20.json \
  --methods balf aa_rank plain_rank \
  --ratios 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99 \
  --calib_size 1024 \
  --seed 0

python plot_factorization.py \
  --input_json results/factorization_analysis_resnet20.json \
  --out_dir plots/factorization_resnet20
