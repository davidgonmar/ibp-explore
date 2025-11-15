#!/usr/bin/env bash
set -e
mkdir -p results
mkdir -p plots/pruning_resnet20_retrain


: '
# no noise
python prune_mi_analysis_retrain.py \
  --model resnet20 \
  --dataset cifar10 \
  --checkpoint resnet20_cifar.pth \
  --out_json results/pruning_analysis_resnet20_retrain.json \
  --methods global_l1 taylor_unstructured \
  --ratios 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99 \
  --calib_size 1024 \
  --seed 0 \
  --retrain_epochs 5 \
  --retrain_lr 1e-3 \
  --retrain_batch_size 128 \
  --retrain_size 5000

python plot_pruning_retrain.py \
  --input_json results/pruning_analysis_resnet20_retrain.json \
  --out_dir plots/pruning_resnet20_retrain
'

# with noise
python prune_mi_analysis_retrain.py \
  --model resnet20 \
  --dataset cifar10 \
  --checkpoint resnet20_cifar.pth \
  --out_json results/pruning_analysis_resnet20_retrain_noise_0_8.json \
  --methods global_l1 \
  --ratios 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99 0.999 \
  --calib_size 1024 \
  --seed 0 \
  --retrain_epochs 10 \
  --retrain_lr 1e-3 \
  --retrain_batch_size 128 \
  --retrain_size 50000 \
  --retrain_label_noise 0.2

python plot_pruning_retrain.py \
    --input_json results/pruning_analysis_resnet20_retrain_noise_0_8.json \
    --out_dir plots/pruning_resnet20_retrain_noise_0_8

