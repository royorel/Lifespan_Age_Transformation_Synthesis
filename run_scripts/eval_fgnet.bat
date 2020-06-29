@echo off

set CUDA_VISIBLE_DEVICES=0

python test_fgnet.py --fgnet --dataroot ../FGNET_with_nvidia_alignment/males --name test_model --which_epoch 400 --how_many 100 --display_id 0 --verbose
