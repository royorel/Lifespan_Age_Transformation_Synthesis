@echo off

set CUDA_VISIBLE_DEVICES=0

python test.py --verbose --dataroot ./datasets/males --name males_model --which_epoch latest --how_many 100 --display_id 0
