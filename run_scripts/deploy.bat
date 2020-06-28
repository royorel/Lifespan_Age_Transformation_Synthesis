@echo off

set CUDA_VISIBLE_DEVICES=0

python test.py --dataroot ./datasets/males --name males_model --which_epoch 400 --sort_order 0-2,3-6,7-9,15-19,30-39,50-69 --display_id 0 --conv_weight_norm --use_modulated_conv --normalize_mlp --deploy --image_path_file males_image_list.txt --full_progression --verbose
