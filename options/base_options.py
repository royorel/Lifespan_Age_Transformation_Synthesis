### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
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
        self.parser.add_argument('--model', type=str, default='flowgan_hd', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance, batch, group or layer normalization')
        self.parser.add_argument('--tex_group_norm', action='store_true', help='try group normalization for texture')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--fgnet', action='store_true', help='are we evaluating the FGNET dataset')
        self.parser.add_argument('--youtube', action='store_true', help='are we evaluating the Youtube dataset')
        self.parser.add_argument('--original_munit', action='store_true', help='run the original MUNIT setup')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='../Combined_Face_Dataset_with_masks_HQ200/males/')
        self.parser.add_argument('--sort_classes', type=bool, default=True, help='a flag that indicates whether to sort the classes')
        self.parser.add_argument('--sort_order', type=str, help='a specific order to sort the classes, must contain all classes, only works when sort_classes is true')
        self.parser.add_argument('--frontal_only', action='store_true', help='when true, only frontal images are used')
        self.parser.add_argument('--use_flow_classes', action='store_true', help='a flag that indicates whether to use predefined classes for flow')
        self.parser.add_argument('--num_flow_classes', type=int, default=6, choices=[2,6,7], help='a flag that indicates how many classes to use for flow')
        self.parser.add_argument('--use_cropped', type=bool, default=True, help='a flag that indicates whether to use cropped version of the images')
        self.parser.add_argument('--use_masks', action='store_true', help='a flag that indicates whether to use image masks')
        self.parser.add_argument('--use_xy', action='store_true', help='a flag that indicates whether to use xy coordinates')
        self.parser.add_argument('--use_flow_layers', action='store_true', help='a flag that indicates whether to use flow prediction layers in the generator')
        self.parser.add_argument('--use_landmarks', action='store_true', help='a flag that indicates whether to use facial landmarks')
        self.parser.add_argument('--embed_landmarks', action='store_true', help='a flag that indicates whether to use an embedding layer for the facial landmarks')
        self.parser.add_argument('--use_parsings', action='store_true', help='a flag that indicates whether to use image parsings')
        self.parser.add_argument('--use_xy_flow_inputs', action='store_true', help='a flag that indicates whether to use xy in the flow network input')
        self.parser.add_argument('--use_rgbxy_flow_inputs', action='store_true', help='a flag that indicates whether to use rgb and xy in the flow network input')
        self.parser.add_argument('--use_parsings_tex_in', action='store_true', help='a flag that indicates whether to use image parsings in the texture network input')
        self.parser.add_argument('--use_parsings_tex_out', action='store_true', help='a flag that indicates whether the texture network will output parsing maps too')
        self.parser.add_argument('--parsing_format', type=str, default='labels', choices=['image', 'labels'], help='the format of the parsing data. can be: labels or image')
        self.parser.add_argument('--parsing_labels_num', type=int, default=19, help='the number of parsing labels to use. can be either 3, 11, 15 or 19.')
        self.parser.add_argument('--mask_gen_outputs', action='store_true', help='if specified, texture part is masked after the texture generator, not before it')
        self.parser.add_argument('--hair_only', action='store_true', help='when true, only hair parsing is used for training')
        self.parser.add_argument('--no_background', action='store_true', help='when true, remove background from images')
        self.parser.add_argument('--no_facial_hair', action='store_true', help='when true, the facial hair label is not used in the flow net')
        self.parser.add_argument('--no_clothing_items', action='store_true', help='when true, clothing items labels (clothing item + hair accesory) are not used in the texture net')
        self.parser.add_argument('--no_neck_tex', action='store_true', help='when true, the neck label is not used in the texture net')
        self.parser.add_argument('--merge_eyes_and_lips', action='store_true', help='when true, the left eye and right eye (as well as eyebrows) and the upper and lower lips will have the same label')
        self.parser.add_argument('--no_facial_features', action='store_true', help='when true, eyes, eyebrows, nose, lips & mouth will have the same label as the face')
        self.parser.add_argument('--downsample_texture', action='store_true', help='when true, inputs to the texture net will be downsampled. This is used when training the inner part of the texture net')
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
        self.parser.add_argument('--gan_mode', type=str, default='texture_only', choices=['flow_only', 'texture_only', 'flow_and_texture','seg_only','seg_and_texture'])
        self.parser.add_argument('--netG', type=str, default='cond_local', help='selects model to use for texture netG')
        self.parser.add_argument('--residual_tex', action='store_true', help='if specified, netG_tex predicts the residual')
        self.parser.add_argument('--use_modulated_conv', action='store_true', help='if specified, use modulated conv layers in the decoder like in StyleGAN2')
        self.parser.add_argument('--conv_weight_norm', action='store_true', help='if specified, use weight normalization in conv and linear layers like in progrssive growing of GANs')
        self.parser.add_argument('--decoder_norm',type=str, default='pixel', choices=['layer','pixel','none'], help='type of upsampling layers normalization')
        self.parser.add_argument('--upsample_norm',type=str, default='adain', choices=['layer','instance', 'adain'], help='type of upsampling layers spatial normalization')
        self.parser.add_argument('--n_adaptive_blocks', type=int, default=4, help='# of adaptive normalization blocks')
        self.parser.add_argument('--last_upconv_out_layers', type=int, default=-1, help='# of output layers in stylegan last upsampling layer')
        self.parser.add_argument('--conv_img_kernel_size', type=int, default=1, help='kernel size of stylegan last upsampling layer')
        self.parser.add_argument('--activation',type=str, default='relu', choices=['relu','lrelu', 'blrelu'], help='type of generator activation layer')
        self.parser.add_argument('--use_tanh', action='store_true', help='if specified, use Tanh in the StyleGAN decoder output')
        self.parser.add_argument('--normalize_mlp', action='store_true', help='if specified, normalize the generator MLP inputs and outputs')
        self.parser.add_argument('--no_moving_avg', action='store_true', help='if specified, do not use moving average network')
        self.parser.add_argument('--truncate_std', action='store_true', help='if specified, use blrelu layer to truncate adain std outputs')
        self.parser.add_argument('--residual_bottleneck', action='store_true', help='if specified, use residual stylegan blocks')
        self.parser.add_argument('--use_resblk_pixel_norm', action='store_true', help='if specified, apply pixel norm on the resnet block outputs')
        self.parser.add_argument('--tex_res_block_const', type=float, default=1.0, help='this const multiply the identity in the resnet block')
        self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        self.parser.add_argument('--keep_G_flow_global', action='store_true', help='if specified, G_flow will not have a local enhancer')
        self.parser.add_argument('--no_rec_flow', action='store_true', help='if specified, do *not* use reconstruction network for flow')
        self.parser.add_argument('--no_rec_tex', action='store_true', help='if specified, do *not* use reconstruction network for texture')
        self.parser.add_argument('--cond_type', type=int, default=0, help='type of condition vector: 0 for one hot, 1 for class number')
        self.parser.add_argument('--no_cond_noise', action='store_true', help='remove gaussian noise from latent age code')
        self.parser.add_argument('--adain_gen_style_dim', type=int, default=100, help='dimension of adain generator style latent code')
        self.parser.add_argument('--adain_one_hot_class_code', action='store_true', help='if specified, decoder input latent code will be a one-hot based vector + noise, otherwise it\'s a guassian with different mean per class')
        self.parser.add_argument('--vae_style_encoder', action='store_true', help='if specified, style encoder will be a VAE')
        self.parser.add_argument('--embed_latent', action='store_true', help='if specified, use MLP to embed gaussian inputs to the style latent space')
        self.parser.add_argument('--parsings_transformation', action='store_true', help='if specified, the texture GAN is applied on the segmentation images')
        self.parser.add_argument('--use_expanded_parsings', action='store_true', help='if specified, use the one-hot representation of the segmentation images')
        self.parser.add_argument('--n_downsample_global_flow', type=int, default=3, help='number of downsampling layers in netG_flow')
        self.parser.add_argument('--n_downsample_global_tex', type=int, default=2, help='number of downsampling layers in netG_tex')
        self.parser.add_argument('--n_downsample_global_seg', type=int, default=2, help='number of downsampling layers in netG_seg')
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--use_cond_resnet_block', action='store_true', help='if specified, insert the local condition before the residual blocks')
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')
        self.parser.add_argument('--use_orig_age_features_within_domain', action='store_true', help='if specified, use_original age features for within domain forward pass of adain generator')

        # for encoder features
        self.parser.add_argument('--use_encoding_net', action='store_true', help='if specified, encode input images features')
        self.parser.add_argument('--use_avg_features', action='store_true', help='if specified, input images features are averaged per label')
        self.parser.add_argument('--feat_num', type=int, default=5, help='vector length for encoded features')
        self.parser.add_argument('--use_encoding_net_flow', action='store_true', help='if specified, encode flow inputs features')
        self.parser.add_argument('--flow_feat_num', type=int, default=6, help='vector length for encoded flow features')

        self.parser.add_argument('--verbose', action='store_true', default = False, help='toggles verbose')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
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

            # self.opt.image_path_list = []
            # for path in temp_paths:
            #     self.opt.image_path_list += [os.path.join(self.opt.dataroot, path)]

        # don't flip images when using parsings
        if self.isTrain and (self.opt.use_parsings or self.opt.parsings_transformation):
            self.opt.no_flip = True

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
