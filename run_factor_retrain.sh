#!/usr/bin/env bash
set -e
mkdir -p results
mkdir -p plots/factorization_resnet20_retrain


python factorization_mi_analysis_retrain.py \
  --model resnet20 \
  --dataset cifar10 \
  --checkpoint resnet20_cifar.pth \
  --out_json results/factorization_analysis_resnet20_retrain.json \
  --methods balf aa_rank plain_rank \
  --ratios 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99 \
  --calib_size 1024 \
  --seed 0 \
  --retrain_epochs 5 \
  --retrain_lr 1e-3 \
  --retrain_batch_size 128 \
  --retrain_size 5000

python plot_factorization.py \
  --input_json results/factorization_analysis_resnet20_retrain.json \
  --out_dir plots/factorization_resnet20_retrain

