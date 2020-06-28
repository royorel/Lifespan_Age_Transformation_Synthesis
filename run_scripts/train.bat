@echo off

set CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py --gpu_ids 0,1,2,3 --dataroot ./datasets/males --name males_model --sort_order 0-2,3-6,7-9,15-19,30-39,50-69 --batchSize 6 --decay_epochs 50,100 --decay_adain_affine_layers --conv_weight_norm --use_modulated_conv --normalize_mlp --verbose
