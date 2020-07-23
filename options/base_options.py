### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='debug', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/males/')
        self.parser.add_argument('--sort_classes', type=bool, default=True, help='a flag that indicates whether to sort the classes')
        self.parser.add_argument('--sort_order', type=str, default='0-2,3-6,7-9,15-19,30-39,50-69', help='a specific order to sort the classes, must contain all classes, only works when sort_classes is true')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=6, help='if positive, display all images in a single visdom web panel with certain number of images per row.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')

        # for generator
        self.parser.add_argument('--use_modulated_conv', type=bool, default=True, help='if specified, use modulated conv layers in the decoder like in StyleGAN2')
        self.parser.add_argument('--conv_weight_norm', type=bool, default=True, help='if specified, use weight normalization in conv and linear layers like in progrssive growing of GANs')
        self.parser.add_argument('--id_enc_norm', type=str, default='pixel', help='instance, pixel normalization')
        self.parser.add_argument('--decoder_norm',type=str, default='pixel', choices=['pixel','none'], help='type of upsampling layers normalization')
        self.parser.add_argument('--n_adaptive_blocks', type=int, default=4, help='# of adaptive normalization blocks')
        self.parser.add_argument('--activation',type=str, default='lrelu', choices=['relu','lrelu'], help='type of generator activation layer')
        self.parser.add_argument('--normalize_mlp', type=bool, default=True, help='if specified, normalize the generator MLP inputs and outputs')
        self.parser.add_argument('--no_moving_avg', action='store_true', help='if specified, do not use moving average network')
        self.parser.add_argument('--use_resblk_pixel_norm', action='store_true', help='if specified, apply pixel norm on the resnet block outputs')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--no_cond_noise', action='store_true', help='remove gaussian noise from latent age code')
        self.parser.add_argument('--gen_dim_per_style', type=int, default=50, help='per class dimension of adain generator style latent code')
        self.parser.add_argument('--n_downsample', type=int, default=2, help='number of downsampling layers in generator')
        self.parser.add_argument('--verbose', action='store_true', default = False, help='toggles verbose')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        try:
            self.opt = self.parser.parse_args()
        except: # solves argparse error in google colab
            self.opt = self.parser.parse_args(args=[])

        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # set class specific sort order
        if self.opt.sort_order is not None:
            order = self.opt.sort_order.split(',')
            self.opt.sort_order = []
            for currName in order:
                self.opt.sort_order += [currName]

        # set decay schedule
        if self.isTrain and self.opt.decay_epochs is not None:
            decay_epochs = self.opt.decay_epochs.split(',')
            self.opt.decay_epochs = []
            for curr_epoch in decay_epochs:
                self.opt.decay_epochs += [int(curr_epoch)]

        # create full image paths in traverse/deploy mode
        if (not self.isTrain) and (self.opt.traverse or self.opt.deploy):
            with open(self.opt.image_path_file,'r') as f:
                # temp_paths = f.read().splitlines()
                self.opt.image_path_list = f.read().splitlines()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:# and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
