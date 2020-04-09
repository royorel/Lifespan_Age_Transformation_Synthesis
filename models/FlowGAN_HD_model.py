### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import torch.nn as nn
import os
import re
import sys
import math
import functools
from collections import OrderedDict
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
import util.util as util
from . import networks
from pdb import set_trace as st

class FlowGANHDModel(BaseModel):
    def name(self):
        return 'FlowGANHDModel'

    def init_loss_filter(self, use_cycle_loss, use_id_loss, use_class_loss_flow, use_class_loss_tex, use_gradient_penalty, adain_losses):
        flags = (True, True, True, use_cycle_loss, use_id_loss, True, True, use_gradient_penalty, True, True,
                 'flow' in self.gan_mode, 'flow' in self.gan_mode, self.use_landmarks and self.json_landmarks,
                 adain_losses, adain_losses, adain_losses)
        def loss_filter(g_gan, g_gan_class, g_gan_feat, g_cycle, g_id, d_real, d_fake, grad_penalty, d_class_real, d_class_fake,
                        minflow, flowTV, landmarksLoss, content_reconst, age_reconst, age_embedding):
            return [l for (l,f) in zip((g_gan, g_gan_class, g_gan_feat, g_cycle, g_id, d_real, d_fake, grad_penalty, d_class_real,
                                        d_class_fake, minflow, flowTV, landmarksLoss, content_reconst, age_reconst, age_embedding),flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.traverse = (not self.isTrain) and opt.traverse
        self.compare_to_trained_outputs = (not self.isTrain) and opt.compare_to_trained_outputs
        if self.compare_to_trained_outputs:
            self.compare_to_trained_class = opt.compare_to_trained_class
            self.trained_class_jump = opt.trained_class_jump

        self.deploy = (not self.isTrain) and opt.deploy
        if not self.isTrain and opt.random_seed != -1:
            torch.manual_seed(opt.random_seed)
            torch.cuda.manual_seed_all(opt.random_seed)
            np.random.seed(opt.random_seed)

        self.nb = opt.batchSize
        self.size = opt.fineSize
        self.ngf = opt.ngf
        self.is_G_local = 'local' in opt.netG
        self.ngf_global = self.ngf
        # if self.is_G_local:
        #     self.ngf_global = self.ngf * (2 ** opt.n_local_enhancers)
        # else:
        #     self.ngf_global = self.ngf

        self.numClasses = opt.numClasses
        self.numFlowClasses = opt.numFlowClasses
        self.use_flow_classes = opt.use_flow_classes
        self.use_parsings = opt.use_parsings
        self.use_masks = opt.use_masks
        self.use_xy = opt.use_xy
        self.use_rgbxy_flow_inputs = opt.use_rgbxy_flow_inputs
        self.use_xy_flow_inputs = opt.use_xy_flow_inputs
        self.use_landmarks = opt.use_landmarks
        self.json_landmarks = opt.json_landmarks
        self.no_background = opt.no_background
        self.no_facial_hair = opt.no_facial_hair
        self.no_clothing_items = opt.no_clothing_items
        self.no_neck_tex = opt.no_neck_tex
        self.mask_gen_outputs = opt.mask_gen_outputs
        self.gan_mode = opt.gan_mode
        self.no_rec_flow = opt.no_rec_flow
        self.no_rec_tex = opt.no_rec_tex
        self.keep_G_flow_global = opt.keep_G_flow_global
        self.use_pretrained_flow = self.isTrain and opt.load_pretrained_flow != '' and self.gan_mode == 'flow_and_texture'
        self.use_pretrained_seg = self.isTrain and opt.load_pretrained_seg != '' and self.gan_mode == 'seg_and_texture'
        self.flow_fixed = self.isTrain and opt.niter_fix_flow == 0 and self.gan_mode == 'flow_and_texture'
        self.seg_fixed = self.isTrain and opt.niter_fix_seg == 0 and self.gan_mode == 'seg_and_texture'
        self.downsample_tex = opt.downsample_texture and self.gan_mode == 'flow_and_texture'
        self.downsampler = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.upsampler = torch.nn.functional.upsample
        self.original_munit = opt.original_munit
        self.vae_style_encoder = opt.vae_style_encoder
        self.parsings_transformation = opt.parsings_transformation
        self.use_expanded_parsings = opt.use_expanded_parsings
        self.parsing_labels_num = opt.parsing_labels_num
        self.use_orig_age_features_within_domain = opt.use_orig_age_features_within_domain
        self.use_moving_avg = 'style' in opt.netG and not opt.no_moving_avg
        self.adain_one_hot_class_code = opt.adain_one_hot_class_code
        self.use_flow_layers = opt.use_flow_layers
        if self.isTrain:
            self.two_discriminators = opt.two_discriminators

        if self.use_flow_classes:
            flow_classes = self.numFlowClasses
        else:
            flow_classes = self.numClasses

        self.cond_type = opt.cond_type
        self.no_cond_noise = opt.no_cond_noise
        # texture condition params
        if self.is_G_local:
            # self.cond_global_length = opt.numClasses * math.floor(opt.ngf*8/opt.numClasses)
            self.cond_global_length = opt.numClasses * math.floor(opt.ngf*4/opt.numClasses)
            self.condG_global_dim = int((self.size / 2) / (2 ** opt.n_downsample_global_tex))
            if self.no_rec_tex:
                self.condG_rec_dim = int(self.size / (2 ** opt.n_downsample_global_tex))
            else:
                self.condG_rec_dim = int(self.size / 4)
            self.cond_local_length = opt.numClasses
            self.condG_local_dim = int(self.size / 2)
            self.cond_length = (self.cond_global_length, self.cond_local_length)
            style_dim = None
        else:
            if 'ada' in opt.netG:
                if self.adain_one_hot_class_code:
                    style_dim = opt.adain_gen_style_dim * self.numClasses
                    self.duplicate = opt.adain_gen_style_dim
                else:
                    style_dim = opt.adain_gen_style_dim
                self.cond_global_length = style_dim #code it as an input option later
                self.cond_length = self.cond_global_length
                self.cond_means = torch.arange(self.numClasses) - ((self.numClasses - 1) / 2)
            else:
                self.cond_global_length = opt.numClasses * math.floor(opt.ngf*4/opt.numClasses)
                self.condG_global_dim = int(self.size / (2 ** opt.n_downsample_global_tex))
                if self.no_rec_tex:
                    self.condG_rec_dim = int(self.size / (2 ** opt.n_downsample_global_tex))
                else:
                    self.condG_rec_dim = int(self.size / 4)
                self.cond_length = self.cond_global_length

                style_dim = None

        # # flow condition params
        # self.cond_global_length_flow = 250 #opt.numClasses * math.floor(opt.ngf*4/opt.numClasses)
        # self.condG_global_dim_flow = int(self.size / (2 ** opt.n_downsample_global))
        # self.condG_rec_dim_flow = int(self.size / 4)
        # self.cond_length_flow = self.cond_global_length_flow

        # new flow condition params  (post Nov 17)
        if self.is_G_local and not self.keep_G_flow_global:
            # self.cond_global_length = opt.numClasses * math.floor(opt.ngf*8/opt.numClasses)
            self.cond_global_length_flow = opt.numFlowClasses * math.floor(opt.ngf*4/opt.numFlowClasses)
            self.condG_global_dim_flow = int((self.size / 2) / (2 ** opt.n_downsample_global_flow))
            if self.no_rec_flow:
                self.condG_rec_dim_flow = int(self.size / (2 ** opt.n_downsample_global_flow))
            else:
                self.condG_rec_dim_flow = int(self.size / 4)
            self.cond_local_length_flow = opt.numClasses
            self.condG_local_dim_flow = int(self.size / 2)
            self.cond_length_flow = (self.cond_global_length, self.cond_local_length)
        else:
            # self.cond_global_length = opt.numFlowClasses * math.floor(opt.ngf*4/opt.numFlowClasses)
            # self.condG_global_dim = int(self.size / (2 ** opt.n_downsample_global))
            # self.condG_rec_dim = int(self.size / 4)
            # self.cond_length = self.cond_global_length
            #pre Nov 17th global flow params
            self.cond_global_length_flow = opt.numFlowClasses * math.floor(opt.ngf*4/opt.numFlowClasses)
            self.condG_global_dim_flow = int(self.size / (2 ** opt.n_downsample_global_flow))
            if self.no_rec_flow:
                self.condG_rec_dim_flow = int(self.size / (2 ** opt.n_downsample_global_flow))
            else:
                self.condG_rec_dim_flow = int(self.size / 4)
            self.cond_length_flow = self.cond_global_length_flow

        self.aux_disc_flow = False
        self.aux_disc_tex = False

        self.tex2flow_mapping = opt.tex2flow_mapping
        self.active_classes_mapping = opt.active_classes_mapping
        self.inv_active_flow_classes_mapping = opt.inv_active_flow_classes_mapping
        if not self.isTrain:
            self.fgnet = opt.fgnet
            self.debug_mode = opt.debug_mode
        else:
            self.fgnet = False
            self.debug_mode = False

        ##### define networks
        # Generators & reconstruction networks
        netG_input_nc = opt.input_nc
        netG_output_nc = opt.output_nc
        ngfR = 64

        self.use_parsings_tex_in = opt.use_parsings_tex_in and self.use_parsings
        self.use_parsings_tex_out = opt.use_parsings_tex_out and self.use_parsings_tex_in
        self.use_cond_resnet_block = opt.use_cond_resnet_block and self.is_G_local

        if 'flow' in self.gan_mode:
            self.use_encoding_net_flow = opt.use_encoding_net_flow
            if opt.netG == 'cond_global' or 'ada' in opt.netG or (opt.netG == 'cond_local' and self.keep_G_flow_global):
                G_flow_type = 'cond_global'
            else:
                G_flow_type = 'cond_local'

            if self.use_rgbxy_flow_inputs:
                netG_flow_input_nc = 8
                netG_flow_output_nc = 2
                netR_flow_input_nc = 8
                netR_flow_output_nc = 2
            elif self.use_xy_flow_inputs:
                netG_flow_input_nc = 5
                netG_flow_output_nc = 2
                netR_flow_input_nc = 5
                netR_flow_output_nc = 2
            elif self.use_encoding_net_flow:
                netG_flow_input_nc = opt.parsing_labels_num + opt.flow_feat_num
                netG_flow_output_nc = 2
                netR_flow_input_nc = netG_input_nc + 3
                netR_flow_output_nc = 2
            elif self.use_landmarks and (not self.json_landmarks) and self.use_parsings:
                netG_flow_input_nc = netG_input_nc + 3
                netG_flow_output_nc = 2
                netR_flow_input_nc = netG_input_nc + 3
                netR_flow_output_nc = 2
            else:
                netG_flow_input_nc = netG_input_nc
                netG_flow_output_nc = 2
                netR_flow_input_nc = netG_input_nc
                netR_flow_output_nc = 2

            self.netG_flow = self.parallelize(networks.define_G(netG_flow_input_nc, netG_flow_output_nc, opt.ngf, G_flow_type, flow_classes, True,
                                              opt.n_downsample_global_flow, opt.n_blocks_global, opt.n_local_enhancers,
                                              opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, cond_length=self.cond_length_flow,
                                              padding_type='zero'))
            if not self.no_rec_flow:
                self.netR_flow = self.parallelize(networks.define_G(netR_flow_input_nc, netR_flow_output_nc, ngfR, 'cond_resnet', is_flow=True,
                                                  numClasses=flow_classes, n_blocks_global=opt.n_blocks_global,
                                                  norm=opt.norm, gpu_ids=self.gpu_ids, cond_length=self.cond_global_length_flow))

            if self.use_encoding_net_flow: #only support both parsings and landmarks
                self.netE_flow = self.parallelize(networks.define_G(6, opt.flow_feat_num, ngfR, 'cond_encoder', numClasses=self.numClasses,
                                                  norm=opt.norm, gpu_ids=self.gpu_ids, cond_length=self.cond_global_length))

        if 'seg' in self.gan_mode:
            # if self.gan_mode == 'seg_and_texture':
            #     self.netG_seg_arch = 'adain_gen'
            # else:
            self.netG_seg_arch = opt.netG
            self.use_encoding_net = False

            netG_seg_input_nc = netG_input_nc
            netG_seg_output_nc = netG_output_nc
            out_type = 'rgb'
            if 'ada' in self.netG_seg_arch and self.use_expanded_parsings:
                netG_seg_input_nc = opt.parsing_labels_num
                netG_seg_output_nc = opt.parsing_labels_num
                #out_type = 'segmentation'

            self.netG_seg = self.parallelize(networks.define_G(netG_seg_input_nc, netG_seg_output_nc, opt.ngf, self.netG_seg_arch, self.numClasses, False,
                                             opt.n_downsample_global_tex, opt.n_blocks_global, opt.n_local_enhancers,
                                             opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, cond_length=self.cond_length,
                                             padding_type='reflect', use_cond_resnet_block=self.use_cond_resnet_block,
                                             style_dim=style_dim, init_type='kaiming', out_type=out_type, conv_weight_norm=opt.conv_weight_norm,
                                             decoder_norm=opt.decoder_norm, upsample_norm=opt.upsample_norm, activation=opt.activation,use_tanh=opt.use_tanh,
                                             truncate_std=opt.truncate_std, adaptive_blocks=opt.n_adaptive_blocks,
                                             use_resblk_pixel_norm=opt.use_resblk_pixel_norm, residual_bottleneck=opt.residual_bottleneck,
                                             last_upconv_out_layers=opt.last_upconv_out_layers, conv_img_kernel_size=opt.conv_img_kernel_size,
                                             normalize_mlp=opt.normalize_mlp, modulated_conv=opt.use_modulated_conv))
            if self.isTrain and self.use_moving_avg:
                self.g_running_seg = networks.define_G(netG_seg_input_nc, netG_seg_output_nc, opt.ngf, self.netG_seg_arch, self.numClasses, False,
                                              opt.n_downsample_global_tex, opt.n_blocks_global, opt.n_local_enhancers,
                                              opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, cond_length=self.cond_length,
                                              padding_type='reflect', use_cond_resnet_block=self.use_cond_resnet_block,
                                              style_dim=style_dim, init_type='kaiming', out_type=out_type, conv_weight_norm=opt.conv_weight_norm,
                                              decoder_norm=opt.decoder_norm, upsample_norm=opt.upsample_norm, activation=opt.activation,use_tanh=opt.use_tanh,
                                              truncate_std=opt.truncate_std, adaptive_blocks=opt.n_adaptive_blocks,
                                              use_resblk_pixel_norm=opt.use_resblk_pixel_norm, residual_bottleneck=opt.residual_bottleneck,
                                              last_upconv_out_layers=opt.last_upconv_out_layers, conv_img_kernel_size=opt.conv_img_kernel_size,
                                              normalize_mlp=opt.normalize_mlp, modulated_conv=opt.use_modulated_conv)
                self.g_running_seg.train(False)
                self.requires_grad(self.g_running_seg, flag=False)
                self.accumulate(self.g_running_seg, self.netG_seg, decay=0)


        if 'texture' in self.gan_mode:
            self.use_encoding_net = opt.use_encoding_net
            self.use_avg_features = opt.use_avg_features
            self.netG_tex_arch = opt.netG
            self.tex_res_block_const = opt.tex_res_block_const
            self.residual_tex = opt.residual_tex

            netG_tex_input_nc = netG_input_nc
            netG_tex_output_nc = netG_output_nc
            netR_tex_input_nc = netG_input_nc
            netR_tex_output_nc = netG_output_nc
            out_type = 'rgb'
            if self.use_parsings_tex_in:
                netG_tex_input_nc = netG_input_nc + 3
                netR_tex_input_nc = netG_input_nc + 3
            if self.use_parsings_tex_out:
                netG_tex_output_nc = netG_output_nc + 3
                netR_tex_output_nc = netG_output_nc + 3
            if self.use_encoding_net:
                netG_tex_input_nc = opt.feat_num
                if self.use_parsings_tex_in:
                    netG_tex_input_nc += opt.parsing_labels_num

            if opt.tex_group_norm:
                tex_norm = 'group'
            else:
                tex_norm = opt.norm
            if self.original_munit:
                self.netG_tex_A = networks.define_G(3,3,64,'adain_gen',n_downsample_global=2,norm=tex_norm,
                                                    padding_type='reflect',style_dim=style_dim, gpu_ids=self.gpu_ids,
                                                    init_type='kaiming', vae_style_encoder=self.vae_style_encoder)
                self.netG_tex_B = networks.define_G(3,3,64,'adain_gen',n_downsample_global=2,norm=tex_norm,
                                                    padding_type='reflect',style_dim=style_dim, gpu_ids=self.gpu_ids,
                                                    init_type='kaiming', vae_style_encoder=self.vae_style_encoder)
            else:
                self.netG_tex = self.parallelize(networks.define_G(netG_tex_input_nc, netG_tex_output_nc, opt.ngf, opt.netG, self.numClasses, False,
                                                 opt.n_downsample_global_tex, opt.n_blocks_global, opt.n_local_enhancers,
                                                 opt.n_blocks_local, tex_norm, gpu_ids=self.gpu_ids, cond_length=self.cond_length,
                                                 padding_type='reflect', use_cond_resnet_block=self.use_cond_resnet_block,
                                                 is_residual=self.residual_tex, res_block_const=self.tex_res_block_const,
                                                 style_dim=style_dim, init_type='kaiming', out_type=out_type,
                                                 conv_weight_norm=opt.conv_weight_norm, decoder_norm=opt.decoder_norm, upsample_norm=opt.upsample_norm,
                                                 activation=opt.activation, use_tanh=opt.use_tanh, truncate_std=opt.truncate_std,
                                                 adaptive_blocks=opt.n_adaptive_blocks, use_resblk_pixel_norm=opt.use_resblk_pixel_norm,
                                                 residual_bottleneck=opt.residual_bottleneck, last_upconv_out_layers=opt.last_upconv_out_layers,
                                                 conv_img_kernel_size=opt.conv_img_kernel_size, normalize_mlp=opt.normalize_mlp, modulated_conv=opt.use_modulated_conv,
                                                 use_flow_layers=self.use_flow_layers))
                if self.isTrain and self.use_moving_avg:
                    self.g_running = networks.define_G(netG_tex_input_nc, netG_tex_output_nc, opt.ngf, opt.netG, self.numClasses, False,
                                                  opt.n_downsample_global_tex, opt.n_blocks_global, opt.n_local_enhancers,
                                                  opt.n_blocks_local, tex_norm, gpu_ids=self.gpu_ids, cond_length=self.cond_length,
                                                  padding_type='reflect', use_cond_resnet_block=self.use_cond_resnet_block,
                                                  is_residual=self.residual_tex, res_block_const=self.tex_res_block_const,
                                                  style_dim=style_dim, init_type='kaiming', out_type=out_type,
                                                  conv_weight_norm=opt.conv_weight_norm, decoder_norm=opt.decoder_norm, upsample_norm=opt.upsample_norm,
                                                  activation=opt.activation, use_tanh=opt.use_tanh, truncate_std=opt.truncate_std,
                                                  adaptive_blocks=opt.n_adaptive_blocks, use_resblk_pixel_norm=opt.use_resblk_pixel_norm,
                                                  residual_bottleneck=opt.residual_bottleneck, last_upconv_out_layers=opt.last_upconv_out_layers,
                                                  conv_img_kernel_size=opt.conv_img_kernel_size, normalize_mlp=opt.normalize_mlp, modulated_conv=opt.use_modulated_conv,
                                                  use_flow_layers=self.use_flow_layers)
                    self.g_running.train(False)
                    self.requires_grad(self.g_running, flag=False)
                    self.accumulate(self.g_running, self.netG_tex, decay=0)
                if not self.no_rec_tex:
                    self.netR_tex = self.parallelize(networks.define_G(netR_tex_input_nc, netG_tex_output_nc, ngfR, 'cond_resnet', is_flow=False,
                                                     numClasses=self.numClasses, n_blocks_global=opt.n_blocks_global,
                                                     norm=tex_norm, gpu_ids=self.gpu_ids, cond_length=self.cond_global_length))

                if self.use_encoding_net:
                    self.netE_tex = self.parallelize(networks.define_G(3, opt.feat_num, ngfR, 'cond_encoder', numClasses=self.numClasses,
                                                     norm=tex_norm, gpu_ids=self.gpu_ids, cond_length=self.cond_global_length))

        self.embed_latent = 'ada' in opt.netG and opt.embed_latent
        if self.embed_latent:
            self.target_embedding_A = networks.MLP(style_dim, style_dim, 128, 4).cuda()
            self.target_embedding_B = networks.MLP(style_dim, style_dim, 128, 4).cuda()
            print(self.target_embedding_A)
            print(self.target_embedding_B)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.output_nc
            self.use_class_loss_flow = not opt.no_class_loss_flow
            self.mse_class_loss_flow = opt.mse_class_loss_flow
            self.use_class_loss_tex = not opt.no_class_loss_tex
            self.add_disc_cond_flow_class = opt.add_disc_cond_flow_class
            self.add_disc_cond_tex_class = opt.add_disc_cond_tex_class
            self.num_init_downsample_flow = opt.num_init_downsample_flow
            self.num_init_downsample_tex = opt.num_init_downsample_tex
            self.selective_class_loss_flow = opt.selective_class_loss_flow
            self.selective_class_loss_tex = opt.selective_class_loss_tex
            self.selective_class_loss_seg = opt.selective_class_loss_seg
            self.selective_class_type_flow = opt.selective_class_type_flow
            self.selective_class_type_tex = opt.selective_class_type_tex
            self.selective_class_type_seg = opt.selective_class_type_seg
            self.getFinalFeat_flow = opt.getFinalFeat_flow
            self.getFinalFeat_tex = opt.getFinalFeat_tex
            self.per_class_netD_tex = opt.netD == 'perclass'
            self.use_parsings_in_disc = opt.use_parsings_in_disc
            self.use_orig_within_domain = opt.use_orig_within_domain
            self.use_background_loss = opt.use_background_loss
            self.orig_age_features_rec_penalty = opt.orig_age_features_rec_penalty

            if 'flow' in self.gan_mode:
                self.netD_tex_arch = None
                if self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                    netD_flow_input_nc = netD_input_nc + 3
                else:
                    netD_flow_input_nc = netD_input_nc

                self.netD_flow = self.parallelize(networks.define_D(netD_flow_input_nc, opt.ndf, 'multiscale', numClasses=flow_classes,
                                                  n_layers_D=opt.n_layers_D, norm=opt.norm, use_sigmoid=use_sigmoid,
                                                  num_D=opt.num_D_flow, num_init_downsample=self.num_init_downsample_flow,
                                                  getIntermFeat=not opt.no_ganIntermFeat, getFinalFeat=self.getFinalFeat_flow,
                                                  use_class_head=self.use_class_loss_flow, selective_class_loss=self.selective_class_loss_flow,
                                                  classify_fakes=self.opt.classify_fakes, use_disc_cond_with_class=self.add_disc_cond_flow_class,
                                                  mse_class_loss=self.mse_class_loss_flow, gpu_ids=self.gpu_ids))
            if self.gan_mode == 'seg_only':
                self.netD_seg_arch = opt.netD
                n_layers_D_seg = opt.n_layers_D

                # if 'ada' in opt.netG and self.use_expanded_parsings:
                #     netD_seg_input_nc = opt.parsing_labels_num
                # else:
                #     netD_seg_input_nc = netD_input_nc

                self.netD_seg = self.parallelize(networks.define_D(netD_input_nc, opt.ndf, opt.netD, numClasses=self.numClasses,
                                                 n_layers_D=n_layers_D_seg, norm=opt.norm, use_sigmoid=use_sigmoid,
                                                 num_D=opt.num_D_seg, num_init_downsample=0,
                                                 getIntermFeat=not opt.no_ganIntermFeat, getFinalFeat=False,
                                                 use_class_head=False, selective_class_loss=self.selective_class_loss_seg,
                                                 classify_fakes=self.opt.classify_fakes, use_disc_cond_with_class=self.add_disc_cond_tex_class,
                                                 use_norm=False, gpu_ids=self.gpu_ids, init_type='kaiming'))

            if 'texture' in self.gan_mode:
                self.netD_tex_arch = opt.netD
                if self.netD_tex_arch == 'aux':
                    if self.downsample_tex:
                        n_layers_D_tex = 5
                    else:
                        n_layers_D_tex = 6 # there was 1 setup where I used 4, suffix is: aux_discriminator_4_D_layers
                else:
                    n_layers_D_tex = opt.n_layers_D

                two_disc_cond = False
                if self.use_parsings_tex_out or self.use_parsings_in_disc:
                    netD_tex_input_nc = netD_input_nc + 3
                elif self.gan_mode == 'seg_and_texture' and self.use_parsings_in_disc:
                    if self.use_expanded_parsings:
                        netD_tex_input_nc = netD_input_nc + opt.parsing_labels_num
                    else:
                        netD_tex_input_nc = netD_input_nc + 3
                elif self.use_flow_layers:
                    one_disc_cond = self.use_flow_layers and (not self.two_discriminators)
                    two_disc_cond = self.use_flow_layers and self.two_discriminators
                    if one_disc_cond:
                        netD_tex_input_nc = netD_input_nc + 3
                    else:
                        netD_tex_input_nc = netD_input_nc
                else:
                    netD_tex_input_nc = netD_input_nc

                if 'ada' in opt.netG:
                    use_norm_D_tex = opt.use_norm_D_tex
                else:
                    use_norm_D_tex = True

                if self.original_munit:
                    self.netD_tex_A = networks.define_D(3, opt.ndf, 'multiscale', numClasses=1, num_D=opt.num_D_tex, n_layers_D=n_layers_D_tex,
                                                        use_norm=use_norm_D_tex, gpu_ids=self.gpu_ids, init_type='kaiming')
                    self.netD_tex_B = networks.define_D(3, opt.ndf, 'multiscale', numClasses=1, num_D=opt.num_D_tex, n_layers_D=n_layers_D_tex,
                                                        use_norm=use_norm_D_tex, gpu_ids=self.gpu_ids, init_type='kaiming')

                else:
                    self.netD_tex = self.parallelize(networks.define_D(netD_tex_input_nc, opt.ndf, opt.netD, numClasses=self.numClasses,
                                                     n_layers_D=n_layers_D_tex, norm=tex_norm, use_sigmoid=use_sigmoid,
                                                     num_D=opt.num_D_tex, num_init_downsample=self.num_init_downsample_tex,
                                                     getIntermFeat=not opt.no_ganIntermFeat, getFinalFeat=self.getFinalFeat_tex,
                                                     use_class_head=self.use_class_loss_tex, selective_class_loss=self.selective_class_loss_tex,
                                                     classify_fakes=self.opt.classify_fakes, use_disc_cond_with_class=self.add_disc_cond_tex_class,
                                                     use_norm=use_norm_D_tex, gpu_ids=self.gpu_ids, init_type='kaiming'))#, activation=opt.activation)
                    if two_disc_cond:
                        self.netD_seg = self.parallelize(networks.define_D(netD_tex_input_nc, opt.ndf, opt.netD, numClasses=self.numClasses,
                                                         n_layers_D=n_layers_D_tex, norm=tex_norm, use_sigmoid=use_sigmoid,
                                                         num_D=opt.num_D_tex, num_init_downsample=self.num_init_downsample_tex,
                                                         getIntermFeat=not opt.no_ganIntermFeat, getFinalFeat=self.getFinalFeat_tex,
                                                         use_class_head=self.use_class_loss_tex, selective_class_loss=self.selective_class_loss_tex,
                                                         classify_fakes=self.opt.classify_fakes, use_disc_cond_with_class=self.add_disc_cond_tex_class,
                                                         use_norm=use_norm_D_tex, gpu_ids=self.gpu_ids, init_type='kaiming'))

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if (not self.isTrain) or (self.isTrain and opt.continue_train) else opt.load_pretrain
            if self.original_munit:
                self.load_network(self.netG_tex_A, 'G_tex_A', opt.which_epoch, pretrained_path)
                self.load_network(self.netG_tex_B, 'G_tex_B', opt.which_epoch, pretrained_path)
                if self.isTrain:
                    self.load_network(self.netD_tex_A, 'D_tex_A', opt.which_epoch, pretrained_path)
                    self.load_network(self.netD_tex_B, 'D_tex_B', opt.which_epoch, pretrained_path)
            else:
                if 'flow' in self.gan_mode:
                    self.load_network(self.netG_flow, 'G_flow', opt.which_epoch, pretrained_path)
                    if not self.no_rec_flow:
                        self.load_network(self.netR_flow, 'R_flow', opt.which_epoch, pretrained_path)
                    if self.use_encoding_net_flow:
                        self.load_network(self.netE_flow, 'E_flow', opt.which_epoch, pretrained_path)
                    if self.isTrain:
                        self.load_network(self.netD_flow, 'D_flow', opt.which_epoch, pretrained_path)
                if 'texture' in self.gan_mode:
                    if self.isTrain:
                        self.load_network(self.netG_tex, 'G_tex', opt.which_epoch, pretrained_path)
                        self.load_network(self.netD_tex, 'D_tex', opt.which_epoch, pretrained_path)
                        if self.use_moving_avg:
                            self.load_network(self.g_running, 'g_running', opt.which_epoch, pretrained_path)
                    elif self.use_moving_avg:
                        self.load_network(self.netG_tex, 'g_running', opt.which_epoch, pretrained_path)
                    else:
                        self.load_network(self.netG_tex, 'G_tex', opt.which_epoch, pretrained_path)
                    if not self.no_rec_tex:
                        self.load_network(self.netR_tex, 'R_tex', opt.which_epoch, pretrained_path)
                    if self.use_encoding_net:
                        self.load_network(self.netE_tex, 'E_tex', opt.which_epoch, pretrained_path)
                if 'seg' in self.gan_mode:
                    if self.isTrain:
                        try:
                            self.load_network(self.netG_seg, 'G_seg', opt.which_epoch, pretrained_path)
                            self.load_network(self.netD_seg, 'D_seg', opt.which_epoch, pretrained_path)
                        except:
                            print('Trying to find a segmentation net saved as net_G_tex.pth')
                            self.load_network(self.netG_seg, 'G_tex', opt.which_epoch, pretrained_path)
                            self.load_network(self.netD_seg, 'D_tex', opt.which_epoch, pretrained_path)
                        if self.use_moving_avg:
                            self.load_network(self.g_running_seg, 'g_running_seg', opt.which_epoch, pretrained_path)
                    elif self.use_moving_avg:
                        self.load_network(self.netG_seg, 'g_running_seg', opt.which_epoch, pretrained_path)
                    else:
                        try:
                            self.load_network(self.netG_seg, 'G_seg', opt.which_epoch, pretrained_path)
                        except:
                            print('Trying to find a segmentation net saved as net_G_tex.pth')
                            self.load_network(self.netG_seg, 'G_tex', opt.which_epoch, pretrained_path)
        elif self.gan_mode == 'flow_and_texture' and self.use_pretrained_flow:
            pretrained_path = opt.load_pretrained_flow
            self.load_network(self.netG_flow, 'G_flow', opt.which_epoch, pretrained_path)
            if not self.no_rec_flow:
                self.load_network(self.netR_flow, 'R_flow', opt.which_epoch, pretrained_path)
            self.load_network(self.netD_flow, 'D_flow', opt.which_epoch, pretrained_path)
        elif self.gan_mode == 'seg_and_texture' and self.use_pretrained_seg:
            pretrained_path = opt.load_pretrained_seg
            try:
                self.load_network(self.netG_seg, 'G_seg', opt.which_epoch, pretrained_path)
                # self.load_network(self.netD_seg, 'D_seg', opt.which_epoch, pretrained_path)
            except: # older networks were saved as a texture network
                print('Trying to find a segmentation net saved as net_G_tex.pth')
                self.load_network(self.netG_seg, 'G_tex', opt.which_epoch, pretrained_path)
                # self.load_network(self.netD_seg, 'D_tex', opt.which_epoch, pretrained_path)


        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                opt.pool_size = 0
                print("Fake Pool Not Implemented for MultiGPU. Setting pool size to 0")

            if 'flow' in self.gan_mode:
                self.flow_fake_pools = []
                for i in range(flow_classes):
                    self.flow_fake_pools += [ImagePool(opt.pool_size)]

            if 'texture' in self.gan_mode:
                self.tex_fake_pools = []
                for i in range(self.numClasses):
                    if opt.netD == 'aux' or 'ada' in opt.netG:
                        self.tex_fake_pools += [ImagePool(0)]
                    else:
                        self.tex_fake_pools += [ImagePool(opt.pool_size)]

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_cycle_loss, not opt.no_id_loss,
                                                     self.use_class_loss_flow, self.use_class_loss_tex, opt.netD == 'aux', 'ada' in opt.netG)

            if 'ada' in opt.netG:
                if self.selective_class_loss_tex:
                    if self.selective_class_type_tex == 'hinge':
                        self.criterionGAN = self.parallelize(networks.SelectiveClassesHingeGANLoss())
                    elif self.selective_class_type_tex == 'non_sat':
                        self.criterionGAN = self.parallelize(networks.SelectiveClassesNonSatGANLoss())
                    else:
                        self.criterionGAN = self.parallelize(networks.SelectiveClassesLSGANLoss(tensor=self.Tensor))
                    self.R1_reg = networks.R1_reg()
                elif self.selective_class_loss_seg:
                    if self.selective_class_type_seg == 'hinge':
                        self.criterionGAN = self.parallelize(networks.SelectiveClassesHingeGANLoss())
                    elif self.selective_class_type_seg == 'non_sat':
                        self.criterionGAN = self.parallelize(networks.SelectiveClassesNonSatGANLoss())
                    else:
                        self.criterionGAN = self.parallelize(networks.SelectiveClassesLSGANLoss(tensor=self.Tensor))
                    self.R1_reg = networks.R1_reg()
                else:
                    self.criterionGAN = self.parallelize(networks.NonSatGANLoss())
                    self.R1_reg = networks.R1_reg()
            else:
                if self.selective_class_loss_flow:
                    if self.selective_class_type_flow == 'hinge':
                        self.criterionGAN_flow = self.parallelize(networks.SelectiveClassesHingeGANLoss())
                    elif self.selective_class_type_flow == 'non_sat':
                        self.criterionGAN = self.parallelize(networks.SelectiveClassesNonSatGANLoss())
                    else:
                        self.criterionGAN_flow = self.parallelize(networks.SelectiveClassesLSGANLoss(tensor=self.Tensor))
                    self.R1_reg = networks.R1_reg()
                else:
                    self.criterionGAN_flow = self.parallelize(networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor))

                if self.selective_class_loss_tex:
                    if self.selective_class_type_tex == 'hinge':
                        self.criterionGAN_tex = self.parallelize(networks.SelectiveClassesHingeGANLoss())
                    elif self.selective_class_type_tex == 'non_sat':
                        self.criterionGAN = self.parallelize(networks.SelectiveClassesNonSatGANLoss())
                    else:
                        self.criterionGAN_tex = self.parallelize(networks.SelectiveClassesLSGANLoss(tensor=self.Tensor))
                    self.R1_reg = networks.R1_reg()
                elif self.netD_tex_arch == 'aux':
                    self.criterionGAN_tex = self.parallelize(networks.WGANLoss())
                    self.gradient_penalty = self.parallelize(networks.gradPenalty())
                else:
                    self.criterionGAN_tex = self.parallelize(networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor))

                if self.opt.classify_fakes:
                    num_tex_classification_labels = self.numClasses+1
                    num_flow_classification_labels = self.numFlowClasses+1
                else:
                    num_tex_classification_labels = self.numClasses
                    num_flow_classification_labels = self.numFlowClasses

                self.flow_aux_loss_type = opt.flow_aux_loss_type
                self.tex_aux_loss_type = opt.tex_aux_loss_type
                self.criterionClass_flow = self.parallelize(networks.AuxLoss(mse_loss=self.mse_class_loss_flow, num_classes=num_flow_classification_labels, class_loss=self.flow_aux_loss_type)) # extra class for fake images
                self.criterionClass_tex = self.parallelize(networks.AuxLoss(num_classes=num_tex_classification_labels, class_loss=self.tex_aux_loss_type)) # extra class for fake images

            self.criterionFeat = self.parallelize(torch.nn.L1Loss())

            self.forward_pass_id_loss = opt.forward_pass_id_loss

            self.age_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.content_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            # self.age_embedding_criterion = networks.GaussianKLDLoss()
            if self.embed_latent:
                self.age_embedding_criterion = self.parallelize(torch.nn.TripletMarginLoss())
            elif self.vae_style_encoder:
                self.age_embedding_criterion = self.parallelize(networks.VAEKLDLoss())
            else:
                self.age_embedding_criterion = None
            # if self.netG_tex_arch == 'adain_gen':
            #     self.criterionCycle = networks.FeatureConsistency()
            #     self.criterionID = networks.FeatureConsistency()
            # else:
            if opt.no_vgg_loss:
                self.criterionCycle = self.parallelize(networks.FeatureConsistency()) #torch.nn.L1Loss()
                self.criterionID = self.parallelize(networks.FeatureConsistency()) #torch.nn.L1Loss()
            else:
                vgg_loss = self.parallelize(networks.VGGLoss(self.gpu_ids))
                self.criterionCycle = vgg_loss
                self.criterionID = vgg_loss
                # self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.criterionMinFlow = self.parallelize(networks.MinFlowLoss(norm='l1', tensor=self.Tensor))
            # self.criterionMinFlow = networks.MinFlowLoss(norm='mse', tensor=self.Tensor)
            self.criterionFlowTV = self.parallelize(networks.FlowTVLoss(isotropic=True, tensor=self.Tensor))
            if self.use_landmarks and self.json_landmarks:
                self.criterionLandmarks = self.parallelize(networks.LandmarkLoss(opt.avgs, opt.stds))

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Class','G_GAN_Feat','G_Cycle','G_ID','D_real', 'D_fake', 'Grad_penalty',
                                               'D_real_class', 'D_fake_class', 'MinFlow', 'FlowTV', 'Landmarks_Loss',
                                               'Content_reconst', 'Age_reconst', 'Age_embedding')

            # initialize optimizers
            self.old_lr = opt.lr
            self.decay_method = opt.decay_method
            if self.original_munit:
                paramsD = list(self.netD_tex_A.parameters()) + list(self.netD_tex_B.parameters())
                paramsG = list(self.netG_tex_A.parameters()) + list(self.netG_tex_B.parameters())
                if self.embed_latent:
                    paramsG += list(self.target_embedding_A.parameters()) + list(self.target_embedding_B.parameters())

                self.optimizer = self.get_optim_alg(opt)
                self.optimizer_G = self.optimizer(paramsG, lr=opt.lr)
                self.optimizer_D = self.optimizer(paramsD, lr=opt.lr)
            else:
                # optimizer G
                paramsG = []
                if opt.niter_fix_global > 0:
                    if self.opt.verbose:
                        print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)

                    #netG
                    if 'flow' in self.gan_mode and not self.use_pretrained_flow:
                        params_dict_G_flow = dict(self.netG_flow.named_parameters())
                        for key, value in params_dict_G_flow.items():
                            if key.startswith('model' + str(opt.n_local_enhancers)):
                                paramsG += [{'params':[value],'lr':opt.lr}]
                            else:
                                paramsG += [{'params':[value],'lr':0.0}]

                    if 'seg' in self.gan_mode and not self.use_pretrained_seg:
                        params_dict_G_seg = dict(self.netG_seg.named_parameters())
                        for key, value in params_dict_G_seg.items():
                            if key.startswith('model' + str(opt.n_local_enhancers)):
                                paramsG += [{'params':[value],'lr':opt.lr}]
                            else:
                                paramsG += [{'params':[value],'lr':0.0}]

                    if 'texture' in self.gan_mode:
                        params_dict_G_tex = dict(self.netG_tex.named_parameters())
                        for key, value in params_dict_G_tex.items():
                            if key.startswith('model' + str(opt.n_local_enhancers)):
                                paramsG += [{'params':[value],'lr':opt.lr}]
                            else:
                                paramsG += [{'params':[value],'lr':0.0}]

                    if self.embed_latent:
                        paramsG += list(self.target_embedding_A.parameters()) + list(self.target_embedding_B.parameters())

                    #netR
                    if not self.no_rec_flow and 'flow' in self.gan_mode and not self.use_pretrained_flow:
                        params_dict_R_flow = dict(self.netR_flow.named_parameters())
                        for key, value in params_dict_R_flow.items():
                            paramsG += [{'params':[value],'lr':opt.lr}]

                    if not self.no_rec_tex and 'texture' in self.gan_mode:
                        params_dict_R_tex = dict(self.netR_tex.named_parameters())
                        for key, value in params_dict_R_tex.items():
                            paramsG += [{'params':[value],'lr':opt.lr}]

                    #netE
                    if self.use_encoding_net_flow and 'flow' in self.gan_mode:
                        params_dict_E_flow = dict(self.netE_flow.named_parameters())
                        for key, value in params_dict_E_flow.items():
                            paramsG += [{'params':[value],'lr':opt.lr}]

                    if self.use_encoding_net and 'texture' in self.gan_mode:
                        params_dict_E_tex = dict(self.netE_tex.named_parameters())
                        for key, value in params_dict_E_tex.items():
                            paramsG += [{'params':[value],'lr':opt.lr}]

                else:
                    if 'flow' in self.gan_mode and not self.use_pretrained_flow:
                        params_dict_G_flow = dict(self.netG_flow.named_parameters())
                        for key, value in params_dict_G_flow.items():
                            decay_cond = ('decoder.mlp' in key)
                            if opt.decay_adain_affine_layers:
                                decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                            if decay_cond:
                                paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                            else:
                                paramsG += [{'params':[value],'lr':opt.lr}]
                        if not self.no_rec_flow:
                            params_dict_R_flow = dict(self.netR_flow.named_parameters())
                            for key, value in params_dict_R_flow.items():
                                decay_cond = ('decoder.mlp' in key)
                                if opt.decay_adain_affine_layers:
                                    decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                                if decay_cond:
                                    paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                                else:
                                    paramsG += [{'params':[value],'lr':opt.lr}]
                        if self.use_encoding_net_flow:
                            params_dict_E_flow = dict(self.netE_flow.named_parameters())
                            for key, value in params_dict_E_flow.items():
                                decay_cond = ('decoder.mlp' in key)
                                if opt.decay_adain_affine_layers:
                                    decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                                if decay_cond:
                                    paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                                else:
                                    paramsG += [{'params':[value],'lr':opt.lr}]
                    if 'seg' in self.gan_mode and not self.use_pretrained_seg:
                        params_dict_G_seg = dict(self.netG_seg.named_parameters())
                        for key, value in params_dict_G_seg.items():
                            decay_cond = ('decoder.mlp' in key)
                            if opt.decay_adain_affine_layers:
                                decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                            if decay_cond:
                                paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                            else:
                                paramsG += [{'params':[value],'lr':opt.lr}]
                    if 'texture' in self.gan_mode:
                        params_dict_G_tex = dict(self.netG_tex.named_parameters())
                        for key, value in params_dict_G_tex.items():
                            decay_cond = ('decoder.mlp' in key)
                            if opt.decay_adain_affine_layers:
                                decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                            if decay_cond:
                                paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                            else:
                                paramsG += [{'params':[value],'lr':opt.lr}]
                        if not self.no_rec_tex:
                            params_dict_R_tex = dict(self.netR_tex.named_parameters())
                            for key, value in params_dict_R_tex.items():
                                decay_cond = ('decoder.mlp' in key)
                                if opt.decay_adain_affine_layers:
                                    decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                                if decay_cond:
                                    paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                                else:
                                    paramsG += [{'params':[value],'lr':opt.lr}]
                        if self.use_encoding_net:
                            params_dict_E_tex = dict(self.netE_tex.named_parameters())
                            for key, value in params_dict_E_tex.items():
                                decay_cond = ('decoder.mlp' in key)
                                if opt.decay_adain_affine_layers:
                                    decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                                if decay_cond:
                                    paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                                else:
                                    paramsG += [{'params':[value],'lr':opt.lr}]

                self.optimizer = self.get_optim_alg(opt)
                self.optimizer_G = self.optimizer(paramsG, lr=opt.lr)
                # self.optimizer_G = torch.optim.Adam(paramsG, lr=opt.lr, betas=(opt.beta1, 0.999))

                # optimizer D
                paramsD = []
                if 'flow' in self.gan_mode and not self.use_pretrained_flow:
                    paramsD += list(self.netD_flow.parameters())
                if 'texture' in self.gan_mode:
                    paramsD += list(self.netD_tex.parameters())
                    if two_disc_cond:
                        paramsD += list(self.netD_seg.parameters())
                if self.gan_mode == 'seg_only':
                    paramsD += list(self.netD_seg.parameters())

                self.optimizer_D = self.optimizer(paramsD, lr=opt.lr)
                # self.optimizer_D = torch.optim.Adam(paramsD, lr=opt.lr, betas=(opt.beta1, 0.999))

    def parallelize(self, model):
        if self.isTrain and len(self.gpu_ids) > 0:
            return networks._CustomDataParallel(model)
            # return nn.DataParallel(model)
        else:
            return model


    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def get_optim_alg(self, opt):
        if opt.alg == 'adam':
            optimizer = functools.partial(torch.optim.Adam, betas=(opt.beta1, opt.beta2), weight_decay=opt.optimizer_wd)
        elif opt.alg == 'rmsprop':
            optimizer = functools.partial(torch.optim.RMSprop, weight_decay=opt.optimizer_wd)
        elif opt.alg == 'sgd':
            optimizer = functools.partial(torch.optim.SGD, momentum=0.9, weight_decay=opt.optimizer_wd)

        return optimizer

    def set_loader_mode(self):
        if self.use_flow_classes and self.gan_mode == 'flow_only':
            return 'uniform_flow'
        else:
            return 'uniform_tex'

    def accumulate(self, model1, model2, decay=0.999):
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        model1_parallel = isinstance(model1, nn.DataParallel)
        model2_parallel = isinstance(model2, nn.DataParallel)

        for k in params1.keys():
            if model2_parallel and not model1_parallel:
                k2 = 'module.' + k
            elif model1_parallel and not model2_parallel:
                k2 = re.sub('module.', '', k)
            else:
                k2 = k
            params1[k].data.mul_(decay).add_(1 - decay, params2[k2].data)

    def encode_input(self, data, mode='train'):
        if mode == 'train':
            input_A = data['A']
            input_B = data['B']
            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                facial_hair_mask_A = data['facial_hair_A']
                facial_hair_mask_B = data['facial_hair_B']
            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                clothing_items_mask_A = data['clothing_items_A']
                clothing_items_mask_B = data['clothing_items_B']
            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                neck_mask_A = data['neck_A']
                neck_mask_B = data['neck_B']
            if ('texture' in self.gan_mode and self.use_encoding_net) or \
               ('flow' in self.gan_mode and self.use_encoding_net_flow) or \
               ('seg' in self.gan_mode and self.use_expanded_parsings):
                expanded_A_parsing = data['expanded_A_parsing']
                expanded_B_parsing = data['expanded_B_parsing']
            if 'flow' in self.gan_mode and self.use_landmarks:
                landmarks_A = data['landmarks_A']
                landmarks_B = data['landmarks_B']

            self.class_A = data['A_class']
            self.flow_class_A = data['flow_A_class']
            self.class_B = data['B_class']
            self.flow_class_B = data['flow_B_class']

            # pytorch 0.4.1
            real_A = input_A[:, :3, :, :]
            real_B = input_B[:, :3, :, :]
            if self.use_parsings and self.use_masks:
                mask_A = input_A[:, 3:4, :, :].expand(-1,3,-1,-1).contiguous()
                mask_B = input_B[:, 3:4, :, :].expand(-1,3,-1,-1).contiguous()
                parsing_A = input_A[:, 4:, :, :]
                parsing_B = input_B[:, 4:, :, :]
            elif self.use_masks:
                mask_A = input_A[:, 3:, :, :].expand(-1,3,-1,-1).contiguous()
                mask_B = input_B[:, 3:, :, :].expand(-1,3,-1,-1).contiguous()
            elif self.use_parsings:
                parsing_A = input_A[:, 3:, :, :]
                parsing_B = input_B[:, 3:, :, :]
                mask_A = ((parsing_A.data > 0).sum(dim=1, keepdim=True) > 0).float().expand(-1,3,-1,-1).contiguous()
                mask_B = ((parsing_B.data > 0).sum(dim=1, keepdim=True) > 0).float().expand(-1,3,-1,-1).contiguous()
            else:
                mask_A = self.Tensor(real_A.size()).fill_(1)
                mask_B = self.Tensor(real_B.size()).fill_(1)

            if len(self.gpu_ids) > 0:
                real_A = real_A
                real_B = real_B
                mask_A = mask_A
                mask_B = mask_B
                if self.use_parsings:
                    parsing_A = parsing_A
                    parsing_B = parsing_B
                    if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                        facial_hair_mask_A = facial_hair_mask_A
                        facial_hair_mask_B = facial_hair_mask_B

                    if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                        clothing_items_mask_A = clothing_items_mask_A
                        clothing_items_mask_B = clothing_items_mask_B

                    if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                        neck_mask_A = neck_mask_A
                        neck_mask_B = neck_mask_B

                    if ('texture' in self.gan_mode and self.use_encoding_net) or \
                       ('flow' in self.gan_mode and self.use_encoding_net_flow) or \
                       ('seg' in self.gan_mode and self.use_expanded_parsings):
                        expanded_A_parsing = expanded_A_parsing
                        expanded_B_parsing = expanded_B_parsing

                if 'flow' in self.gan_mode and self.use_landmarks:
                    landmarks_A = landmarks_A
                    landmarks_B = landmarks_B

            # rescale masks to [0 1] values
            if self.use_masks:
                mask_A = (mask_A + 1) / 2
                mask_B = (mask_B + 1) / 2

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                facial_hair_mask_A = (facial_hair_mask_A + 1) / 2
                facial_hair_mask_B = (facial_hair_mask_B + 1) / 2

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                clothing_items_mask_A = (clothing_items_mask_A + 1) / 2
                clothing_items_mask_B = (clothing_items_mask_B + 1) / 2

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                neck_mask_A = (neck_mask_A + 1) / 2
                neck_mask_B = (neck_mask_B + 1) / 2

            if ('texture' in self.gan_mode and self.use_encoding_net) or \
               ('flow' in self.gan_mode and self.use_encoding_net_flow): #or \
               #(self.netG_tex_arch == 'adain_gen' and self.parsings_transformation and self.use_expanded_parsings):
                expanded_A_parsing = (expanded_A_parsing + 1) / 2
                expanded_B_parsing = (expanded_B_parsing + 1) / 2

            self.reals = torch.cat((real_A, real_B), 0).cuda()
            self.masks = torch.cat((mask_A, mask_B), 0).cuda()

            if self.use_xy or self.use_rgbxy_flow_inputs or self.use_flow_layers:
                bSize, ch, h, w = self.reals.size()
                # create original grid (equivalent to numpy meshgrid)
                x = torch.linspace(-1, 1, steps=w).type_as(self.reals)
                y = torch.linspace(-1, 1, steps=h).type_as(self.reals)

                # pytorch 0.4.1
                xx = x.view(1, -1).repeat(bSize, 1, h, 1)
                yy = y.view(-1, 1).repeat(bSize, 1, 1, w)

                self.xy = torch.cat([xx, yy], 1).cuda()
            else:
                self.xy = None

            if self.use_parsings:
                self.parsings = torch.cat((parsing_A, parsing_B), 0).cuda()
                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                    self.facial_hair_masks = torch.cat((facial_hair_mask_A, facial_hair_mask_B), 0).cuda()
                else:
                    self.facial_hair_masks = None

                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                    self.clothing_items_masks = torch.cat((clothing_items_mask_A, clothing_items_mask_B), 0).cuda()
                else:
                    self.clothing_items_masks = None

                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                    self.neck_masks = torch.cat((neck_mask_A, neck_mask_B), 0).cuda()
                else:
                    self.neck_masks = None

            else:
                self.parsings = None
                self.facial_hair_masks = None
                self.neck_masks = None

            if ('texture' in self.gan_mode and self.use_encoding_net) or \
               ('flow' in self.gan_mode and self.use_encoding_net_flow) or \
               ('seg' in self.gan_mode and self.use_expanded_parsings):
                self.expanded_parsings = torch.cat((expanded_A_parsing, expanded_B_parsing), 0).cuda()
            else:
                self.expanded_parsings = None

            if 'flow' in self.gan_mode and self.use_landmarks:
                self.landmarks = torch.cat((landmarks_A, landmarks_B), 0).cuda()
            else:
                self.landmarks = None

        else:
            inputs = data['Imgs']
            if inputs.dim() > 4:
                inputs = inputs.squeeze(0)

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                facial_hair_masks = data['facial_hair']
                if facial_hair_masks.dim() > 4:
                    facial_hair_masks = facial_hair_masks.squeeze(0)

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                clothing_items_masks = data['clothing_items']
                if clothing_items_masks.dim() > 4:
                    clothing_items_masks = clothing_items_masks.squeeze(0)

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                neck_masks = data['neck']
                if neck_masks.dim() > 4:
                    neck_masks = neck_masks.squeeze(0)

            if ('texture' in self.gan_mode and self.use_encoding_net) or \
               ('flow' in self.gan_mode and self.use_encoding_net_flow) or \
               ('seg' in self.gan_mode and self.use_expanded_parsings):
                expanded_parsings = data['expanded_parsings']

            if 'flow' in self.gan_mode and self.use_landmarks:
                landmarks = data['landmarks']

            self.class_A = data['Classes']
            if self.class_A.dim() > 1:
                self.class_A = self.class_A.squeeze(0)

            self.flow_class_A = data['flow_Classes']
            if self.flow_class_A.dim() > 1:
                self.flow_class_A = self.flow_class_A.squeeze(0)

            if torch.is_tensor(data['Valid']):
                self.valid = data['Valid'].bool()
            else:
                self.valid = torch.ones(1, dtype=torch.bool)

            if self.valid.dim() > 1:
                self.valid = self.valid.squeeze(0)

            # self.class_A.resize_(classes.size()).copy_(classes)
            # self.valid.resize_(valid.size()).copy_(valid)
            if isinstance(data['Paths'][0], tuple):
                self.image_paths = [path[0] for path in data['Paths']]
            else:
                self.image_paths = data['Paths']

            self.isEmpty = False if any(self.valid) else True
            if not self.isEmpty:
                available_idx = torch.arange(len(self.class_A))
                select_idx = torch.masked_select(available_idx, self.valid).long()
                inputs = torch.index_select(inputs, 0, select_idx)
                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                    facial_hair_masks = torch.index_select(facial_hair_masks, 0, select_idx)
                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                    clothing_items_masks = torch.index_select(clothing_items_masks, 0, select_idx)
                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                    neck_masks = torch.index_select(neck_masks, 0, select_idx)
                if (('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.use_encoding_net) or \
                   ('flow' in self.gan_mode and self.use_encoding_net_flow) or \
                   ('seg' in self.gan_mode and self.use_expanded_parsings):
                    expanded_parsings = torch.index_select(expanded_parsings, 0, select_idx)
                if 'flow' in self.gan_mode and self.use_landmarks:
                    landmarks = torch.index_select(landmarks, 0, select_idx)

                self.class_A = torch.index_select(self.class_A, 0, select_idx)
                self.flow_class_A = torch.index_select(self.flow_class_A, 0, select_idx)
                self.image_paths = [val for i, val in enumerate(self.image_paths) if self.valid[i] == 1]

            # pytorch 0.4.1
            real_A = inputs[:, :3, :, :]
            if self.use_parsings and self.use_masks:
                mask_A = inputs[:, 3:4, :, :].expand(-1,3,-1,-1).contiguous()
                parsing_A = inputs[:, 4:, :, :]
            elif self.use_masks:
                mask_A = inputs[:, 3:, :, :].expand(-1,3,-1,-1).contiguous()
            elif self.use_parsings:
                parsing_A = inputs[:, 3:, :, :]
                mask_A = ((parsing_A.data > 0).sum(dim=1, keepdim=True) > 0).float().expand(-1,3,-1,-1).contiguous()
            else:
                mask_A = self.Tensor(real_A.size()).fill_(1)

            if len(self.gpu_ids) > 0:
                real_A = real_A
                mask_A = mask_A
                if self.use_parsings:
                    parsing_A = parsing_A
                    if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                        facial_hair_masks = facial_hair_masks

                    if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                        clothing_items_masks = clothing_items_masks

                    if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                        neck_masks = neck_masks

                if ('texture' in self.gan_mode and self.use_encoding_net) or \
                   ('flow' in self.gan_mode and self.use_encoding_net_flow) or \
                   ('seg' in self.gan_mode and self.use_expanded_parsings):
                    expanded_parsings = expanded_parsings

                if 'flow' in self.gan_mode and self.use_landmarks:
                    landmarks = landmarks

            # rescale masks to [0 1] values
            if self.use_masks:
                mask_A = (mask_A + 1) / 2

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                facial_hair_masks = (facial_hair_masks + 1) / 2

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                clothing_items_masks = (clothing_items_masks + 1) / 2

            if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                neck_masks = (neck_masks + 1) / 2

            if ('texture' in self.gan_mode and self.use_encoding_net) or \
               ('flow' in self.gan_mode and self.use_encoding_net_flow): #or \
               #(self.netG_tex_arch == 'adain_gen' and self.parsings_transformation and self.use_expanded_parsings):
                expanded_parsings = (expanded_parsings + 1) / 2

            self.reals = real_A.cuda()
            self.masks = mask_A.cuda()

            if self.use_xy or self.use_rgbxy_flow_inputs:
                bSize, ch, h, w = self.reals.size()
                # create original grid (equivalent to numpy meshgrid)
                x = torch.linspace(-1, 1, steps=w).type_as(self.reals)
                y = torch.linspace(-1, 1, steps=h).type_as(self.reals)

                # pytorch 0.4.1
                xx = x.view(1, -1).repeat(bSize, 1, h, 1)
                yy = y.view(-1, 1).repeat(bSize, 1, 1, w)

                self.xy = torch.cat([xx, yy], 1).cuda()
            else:
                self.xy = None

            if self.use_parsings:
                self.parsings = parsing_A.cuda()
                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_facial_hair:
                    self.facial_hair_masks = facial_hair_masks.cuda()
                else:
                    self.facial_hair_masks = None

                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_clothing_items:
                    self.clothing_items_masks = clothing_items_masks.cuda()
                else:
                    self.clothing_items_masks = None

                if ('texture' in self.gan_mode or 'seg' in self.gan_mode) and self.no_neck_tex:
                    self.neck_masks = neck_masks.cuda()
                else:
                    self.neck_masks = None

            else:
                self.parsings = None
                self.facial_hair_masks = None
                self.neck_masks = None

            if ('texture' in self.gan_mode and self.use_encoding_net) or \
               ('flow' in self.gan_mode and self.use_encoding_net_flow) or \
               ('seg' in self.gan_mode and self.use_expanded_parsings):
                self.expanded_parsings = expanded_parsings.cuda()
            else:
                self.expanded_parsings = None

            if 'flow' in self.gan_mode and self.use_landmarks:
                self.landmarks = landmarks.cuda()
            else:
                self.landmarks = None

        self.masked_reals = self.reals * self.masks


    def get_conditions(self, mode='train'):
        if mode == 'train':
            nb = self.masked_reals.shape[0] // 2
        elif self.traverse or self.deploy:
            if self.traverse and self.compare_to_trained_outputs:
                nb = 2
            else:
                nb = self.numClasses
        else:
            nb = self.numValid

        is_adain_gen = 'seg' in self.gan_mode and 'ada' in self.netG_seg_arch or \
                       'texture' in self.gan_mode and 'ada' in self.netG_tex_arch
        #tex condition mapping
        if 'texture' in self.gan_mode or 'seg' in self.gan_mode:
            if is_adain_gen:
                condG_A_global = self.Tensor(nb, self.cond_global_length)
                condG_B_global = self.Tensor(nb, self.cond_global_length)
                condG_A_orig = self.Tensor(nb, self.cond_global_length)
                condG_B_orig = self.Tensor(nb, self.cond_global_length)
            else:
                condG_A_global = self.Tensor(nb, self.cond_global_length, self.condG_global_dim, self.condG_global_dim)
                condG_B_global = self.Tensor(nb, self.cond_global_length, self.condG_global_dim, self.condG_global_dim)
                if self.is_G_local:
                    condG_A_local = self.Tensor(nb, self.cond_local_length, self.condG_local_dim, self.condG_local_dim)
                    condG_B_local = self.Tensor(nb, self.cond_local_length, self.condG_local_dim, self.condG_local_dim)

            if mode == 'train':
                condD_A = self.Tensor(nb, self.numClasses, self.size, self.size)
                condD_B = self.Tensor(nb, self.numClasses, self.size, self.size)

            # in this case we first reset the condition vectors
            if self.cond_type == 0 or self.cond_type == 2:
                if is_adain_gen:
                    pass
                else:
                    condG_A_global.fill_(-1)
                    condG_B_global.fill_(-1)
                    if self.is_G_local:
                        condG_A_local.fill_(-1)
                        condG_B_local.fill_(-1)
                if mode == 'train':
                    condD_A.fill_(-1)
                    condD_B.fill_(-1)

        #flow condition mapping
        if 'flow' in self.gan_mode:
            flow_condG_A_global = self.Tensor(nb, self.cond_global_length_flow, self.condG_global_dim_flow, self.condG_global_dim_flow)
            flow_condG_B_global = self.Tensor(nb, self.cond_global_length_flow, self.condG_global_dim_flow, self.condG_global_dim_flow)
            if self.is_G_local and not self.keep_G_flow_global:
                flow_condG_A_local = self.Tensor(nb, self.cond_local_length, self.condG_local_dim, self.condG_local_dim)
                flow_condG_B_local = self.Tensor(nb, self.cond_local_length, self.condG_local_dim, self.condG_local_dim)
            if mode == 'train':
                flow_condD_A = self.Tensor(nb, self.numFlowClasses, self.size, self.size)
                flow_condD_B = self.Tensor(nb, self.numFlowClasses, self.size, self.size)

            if self.cond_type == 0 or self.cond_type == 2:
                flow_condG_A_global.fill_(-1)
                flow_condG_B_global.fill_(-1)
                if self.is_G_local and not self.keep_G_flow_global:
                    flow_condG_A_local.fill_(-1)
                    flow_condG_B_local.fill_(-1)
                if mode == 'train':
                    flow_condD_A.fill_(-1)
                    flow_condD_B.fill_(-1)

        if self.no_cond_noise:
            noise_sigma = 0
        else:
            noise_sigma = 0.2

        for i in range(nb):
            if self.cond_type == 0:
                if 'texture' in self.gan_mode or 'seg' in self.gan_mode:
                    if is_adain_gen:
                        if self.adain_one_hot_class_code:
                            condG_A_global[i, :] = (noise_sigma * torch.randn(1, self.cond_global_length)).cuda()
                            condG_A_global[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1
                            if not (self.traverse or self.deploy):
                                condG_B_global[i, :] = (noise_sigma * torch.randn(1, self.cond_global_length)).cuda()
                                condG_B_global[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                                condG_A_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_global_length)).cuda()
                                condG_A_orig[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                                condG_B_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_global_length)).cuda()
                                condG_B_orig[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1
                        else:
                            condG_A_global[i, :] = (noise_sigma * torch.randn(1, self.cond_global_length) + self.class_B[i].float()).cuda()
                            if not (self.traverse or self.deploy):
                                condG_B_global[i, :] = (noise_sigma * torch.randn(1, self.cond_global_length) + self.class_A[i].float()).cuda()
                                condG_A_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_global_length) + self.class_A[i].float()).cuda()
                                condG_B_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_global_length) + self.class_B[i].float()).cuda()
                            # condG_A_global[i, :] = (0.2 * torch.randn(1, self.cond_global_length) + self.cond_means[self.class_B[i]].float()).cuda()
                            # condG_B_global[i, :] = (0.2 * torch.randn(1, self.cond_global_length) + self.cond_means[self.class_A[i]].float()).cuda()
                    else:
                        duplicate = math.floor(self.ngf_global*4/self.numClasses)
                        # Condition G for source A is a one hot at channels target class[i] dedicated channels,
                        # which are class_B[i]*duplicate:(class_B[i] + 1)*duplicate. Rest of the classes are -1
                        condG_A_global[i, self.class_B[i]*duplicate:(self.class_B[i] + 1)*duplicate, :, :] = 1
                        # Condition G for source B is a one hot at channels target class[i] dedicated channels,
                        # which are class_A[i]*duplicate:(class_A[i] + 1)*duplicate. Rest of the classes are -1
                        if not (self.traverse or self.deploy):
                            condG_B_global[i, self.class_A[i]*duplicate:(self.class_A[i] + 1)*duplicate, :, :] = 1
                        # D conditions are opposite from G conditions because the generator output should belong to the target class.
                        # i.e. netG(real_A,condG_A) produces fake_B therefore we should do netD(fake_B,condD_B)
                        if self.is_G_local:
                            condG_A_local[i, self.class_B[i]:(self.class_B[i] + 1), :, :] = 1
                            if not (self.traverse or self.deploy):
                                condG_B_local[i, self.class_A[i]:(self.class_A[i] + 1), :, :] = 1

                    if mode == 'train':
                        condD_A[i, self.class_A[i], :, :] = 1
                        condD_B[i, self.class_B[i], :, :] = 1

                if 'flow' in self.gan_mode:
                    flow_duplicate = math.floor(self.ngf*4/self.numFlowClasses)
                    # Condition G for source A is a one hot at channels target class[i] dedicated channels,
                    # which are class_B[i]*duplicate:(class_B[i] + 1)*duplicate. Rest of the classes are -1
                    flow_condG_A_global[i, self.flow_class_B[i]*flow_duplicate:(self.flow_class_B[i] + 1)*flow_duplicate, :, :] = 1
                    # Condition G for source B is a one hot at channels target class[i] dedicated channels,
                    # which are class_A[i]*duplicate:(class_A[i] + 1)*duplicate. Rest of the classes are -1
                    if not (self.traverse or self.deploy):
                        flow_condG_B_global[i, self.flow_class_A[i]*flow_duplicate:(self.flow_class_A[i] + 1)*flow_duplicate, :, :] = 1
                    # D conditions are opposite from G conditions because the generator output should belong to the target class.
                    # i.e. netG(real_A,condG_A) produces fake_B therefore we should do netD(fake_B,condD_B)
                    if self.is_G_local and not self.keep_G_flow_global:
                        flow_condG_A_local[i, self.flow_class_B[i]:(self.flow_class_B[i] + 1), :, :] = 1
                        if not self.traverse:
                            flow_condG_B_local[i, self.flow_class_A[i]:(self.flow_class_A[i] + 1), :, :] = 1
                    if mode == 'train' and not self.aux_disc_flow: #(self.aux_disc_flow and self.aux_disc_tex):
                        flow_condD_A[i, self.flow_class_A[i]] = 1
                        if not (self.traverse or self.deploy):
                            flow_condD_B[i, self.flow_class_B[i]] = 1

            else: # in this case we just fill a volume with the target class number
                if 'texture' in self.gan_mode or 'seg' in self.gan_mode:
                    if is_adain_gen:
                        pass
                    else:
                        condG_A_global[i, :, :, :].fill_(self.class_B[i])
                        if not (self.traverse or self.deploy):
                            condG_B_global[i, :, :, :].fill_(self.class_A[i])
                        if self.is_G_local:
                            condG_A_local[i, :, :, :].fill_(self.class_B[i])
                            if not (self.traverse or self.deploy):
                                condG_B_local[i, :, :, :].fill_(self.class_A[i])

                    if mode == 'train':
                        condD_A[i, :, :, :].fill_(self.class_A[i])
                        condD_B[i, :, :, :].fill_(self.class_B[i])

                if 'flow' in self.gan_mode:
                    flow_condG_A_global[i, :, :, :].fill_(self.flow_class_B[i])
                    if not (self.traverse or self.deploy):
                        flow_condG_B_global[i, :, :, :].fill_(self.flow_class_A[i])
                    if self.is_G_local and not self.keep_G_flow_global:
                        flow_condG_A_local[i, :, :, :].fill_(self.flow_class_B[i])
                        if not (self.traverse or self.deploy):
                            flow_condG_B_local[i, :, :, :].fill_(self.flow_class_A[i])
                    if mode == 'train' and not self.aux_disc_flow: #(self.aux_disc_flow and self.aux_disc_tex):
                        flow_condD_A[i, :, :, :].fill_(self.flow_class_A[i])
                        flow_condD_B[i, :, :, :].fill_(self.flow_class_B[i])

        if mode == 'train':
            if 'texture' in self.gan_mode or 'seg' in self.gan_mode:
                if is_adain_gen:
                    self.gen_conditions =  torch.cat((condG_A_global, condG_B_global), 0) #torch.cat((self.class_B, self.class_A), 0)
                    self.rec_conditions = torch.cat((condG_B_global, condG_A_global), 0)
                    self.orig_conditions = torch.cat((condG_A_orig, condG_B_orig),0)
                    # self.orig_age_sigma = 0.2
                    self.orig_age_mean = torch.cat((self.class_A.float(), self.class_B.float()), 0).cuda()

                else:
                    ratio = int(self.condG_rec_dim/self.condG_global_dim)
                    self.gen_conditions = torch.cat((condG_A_global, condG_B_global), 0)
                    self.rec_conditions = torch.cat((condG_B_global, condG_A_global), 0).repeat(1,1,ratio,ratio)
                    if self.is_G_local:
                        self.gen_conditions = (self.gen_conditions, torch.cat((condG_A_local, condG_B_local), 0))

                self.real_disc_conditions = torch.cat((condD_A, condD_B), 0)
                self.fake_disc_conditions = torch.cat((condD_B, condD_A), 0)

            if 'flow' in self.gan_mode:
                ratio_flow = int(self.condG_rec_dim_flow/self.condG_global_dim_flow)
                self.flow_gen_conditions = torch.cat((flow_condG_A_global, flow_condG_B_global), 0)
                self.flow_rec_conditions = torch.cat((flow_condG_B_global, flow_condG_A_global), 0).repeat(1,1,ratio_flow,ratio_flow)
                if self.is_G_local and not self.keep_G_flow_global:
                    self.flow_gen_conditions = (self.flow_gen_conditions, torch.cat((flow_condG_A_local, flow_condG_B_local), 0))

                self.flow_real_disc_conditions = torch.cat((flow_condD_A, flow_condD_B), 0)
                self.flow_fake_disc_conditions = torch.cat((flow_condD_B, flow_condD_A), 0)
        else:
            if 'texture' in self.gan_mode or 'seg' in self.gan_mode:
                if is_adain_gen:
                    self.gen_conditions = condG_A_global #self.class_B
                    if not (self.traverse or self.deploy):
                        self.rec_conditions = condG_B_global #self.class_A
                        self.orig_conditions = condG_A_orig
                else:
                    ratio = int(self.condG_rec_dim/self.condG_global_dim)
                    self.gen_conditions = condG_A_global
                    if not (self.traverse or self.deploy):
                        self.rec_conditions = condG_B_global.repeat(1,1,ratio,ratio)
                    if self.is_G_local:
                        self.gen_conditions = (self.gen_conditions, condG_A_local)

            if 'flow' in self.gan_mode:
                ratio_flow = int(self.condG_rec_dim_flow/self.condG_global_dim_flow)
                self.flow_gen_conditions = flow_condG_A_global
                if not (self.traverse or self.deploy):
                    self.flow_rec_conditions = flow_condG_B_global.repeat(1,1,ratio_flow,ratio_flow)
                if self.is_G_local and not self.keep_G_flow_global:
                    self.flow_gen_conditions = (self.flow_gen_conditions, flow_condG_A_local)


    def discriminate(self, input_images, use_pool=False):
        flow_input_images, tex_input_images = input_images[0], input_images[1]
        if 'flow' in self.gan_mode and not self.flow_fixed:
            bSize = int(flow_input_images.size(0)/2)
            flow_input_im_A = flow_input_images[:bSize,:,:,:].detach()
            flow_input_im_B = flow_input_images[bSize:,:,:,:].detach()
            if use_pool:
                flow_fake_query_A = torch.zeros_like(flow_input_im_A)
                flow_fake_query_B = torch.zeros_like(flow_input_im_B)
                for i in range(bSize):
                    flow_cls_A = self.flow_class_A[i].item()
                    flow_cls_B = self.flow_class_B[i].item()
                    # when we use the image pool input_im_A should belong to class B
                    # and input_im_B should belong to class A
                    flow_fake_query_B[i] = self.flow_fake_pools[flow_cls_B].query(flow_input_im_A[i])
                    flow_fake_query_A[i] = self.flow_fake_pools[flow_cls_A].query(flow_input_im_B[i])

                flow_fake_query = torch.cat((flow_fake_query_B,flow_fake_query_A), 0)
                if (self.use_class_loss_flow and (not self.add_disc_cond_flow_class)) or self.selective_class_loss_flow:
                    flow_disc_input = flow_fake_query
                else:
                    flow_disc_input = torch.cat((flow_fake_query, self.flow_fake_disc_conditions), 1)
            else:
                flow_query = torch.cat((flow_input_im_A,flow_input_im_B), 0)
                if self.use_class_loss_flow and (not self.add_disc_cond_flow_class):
                    flow_disc_input = flow_query
                else:
                    flow_disc_input = torch.cat((flow_query, self.flow_real_disc_conditions), 1)

            pred_flow_gan, pred_flow_class, pred_flow_feat = self.netD_flow(flow_disc_input)
        else:
            pred_flow_gan, pred_flow_class, pred_flow_feat = None, None, None

        if 'texture' in self.gan_mode:
            bSize = int(tex_input_images.size(0)/2)
            tex_input_im_A = tex_input_images[:bSize,:,:,:].detach()
            tex_input_im_B = tex_input_images[bSize:,:,:,:].detach()
            if use_pool:
                if self.netD_tex_arch == 'aux':
                    # here the outputs are aligned for the gradient penalty to work properly
                    # (the interpolated images should be between inputs and outputs)
                    tex_disc_input = torch.cat((tex_input_im_A,tex_input_im_B), 0)
                else:
                    tex_fake_query_A = torch.zeros_like(tex_input_im_A)
                    tex_fake_query_B = torch.zeros_like(tex_input_im_B)
                    for i in range(bSize):
                        cls_A = self.class_A[i].item()
                        cls_B = self.class_B[i].item()
                        # when we use the image pool input_im_A should belong to class B
                        # and input_im_B should belong to class A
                        tex_fake_query_B[i] = self.tex_fake_pools[cls_B].query(tex_input_im_A[i])
                        tex_fake_query_A[i] = self.tex_fake_pools[cls_A].query(tex_input_im_B[i])

                    tex_fake_query = torch.cat((tex_fake_query_B,tex_fake_query_A), 0)
                    if (self.use_class_loss_tex and (not self.add_disc_cond_tex_class)) or self.selective_class_loss_tex or self.per_class_netD_tex:
                        tex_disc_input = tex_fake_query
                    else:
                        tex_disc_input = torch.cat((tex_fake_query, self.fake_disc_conditions), 1)

                if self.per_class_netD_tex:
                    disc_classes = torch.cat((self.class_B,self.class_A),0).tolist()

            else:
                if self.netD_tex_arch == 'aux':
                    tex_disc_input = torch.cat((tex_input_im_A,tex_input_im_B), 0)
                else:
                    tex_query = torch.cat((tex_input_im_A,tex_input_im_B), 0)
                    if (self.use_class_loss_tex and (not self.add_disc_cond_tex_class)) or self.selective_class_loss_tex or self.per_class_netD_tex:
                        tex_disc_input = tex_query
                    else:
                        tex_disc_input = torch.cat((tex_query, self.real_disc_conditions), 1)

                if self.per_class_netD_tex:
                    disc_classes = torch.cat((self.class_A,self.class_B),0).tolist()

            if self.per_class_netD_tex:
                pred_tex_gan, pred_tex_class, pred_tex_feat = self.netD_tex(tex_disc_input,disc_classes)
            else:
                pred_tex_gan, pred_tex_class, pred_tex_feat = self.netD_tex(tex_disc_input)
        else:
            pred_tex_gan, pred_tex_class, pred_tex_feat = None, None, None

        return pred_flow_gan, pred_flow_class, pred_flow_feat, pred_tex_gan, pred_tex_class, pred_tex_feat

    def update_G(self, infer=False):
        self.optimizer_G.zero_grad()
        if self.original_munit:
            # remove clothing items from texture input
            if self.no_background and not self.mask_gen_outputs:
                gen_in = self.masked_reals
            else:
                gen_in = self.reals

            if self.no_clothing_items:
                gen_in = gen_in * (1-self.clothing_items_masks)
                self.masks = self.masks * (1-self.clothing_items_masks)

            if self.no_neck_tex:
                gen_in = gen_in * (1-self.neck_masks)
                self.masks = self.masks * (1-self.neck_masks)

            b = self.class_A.shape[0]
            if self.class_A == 0:
                gen_in_A = gen_in[:b,:,:,:]
                gen_in_B = gen_in[b:,:,:,:]
                mask_A = self.masks[:b,:,:,:]
                mask_B = self.masks[b:,:,:,:]
            else:
                gen_in_A = gen_in[b:,:,:,:]
                gen_in_B = gen_in[:b,:,:,:]
                mask_A = self.masks[b:,:,:,:]
                mask_B = self.masks[:b,:,:,:]

            #generate random gaussians
            z_A = torch.randn(b, self.cond_global_length).cuda()
            z_B = torch.randn(b, self.cond_global_length).cuda()
            #embed gaussians into target style space
            if self.embed_latent:
                target_A = self.target_embedding_A(z_A)
                target_B = self.target_embedding_B(z_B)
            else:
                target_A = z_A
                target_B = z_B

            #encode images
            orig_id_features_A, orig_age_encoding_A = self.netG_tex_A.encode(gen_in_A, mask_A)
            orig_id_features_B, orig_age_encoding_B = self.netG_tex_B.encode(gen_in_B, mask_B)
            if self.vae_style_encoder:
                mu_A, logvar_A = orig_age_encoding_A[:,0], orig_age_encoding_A[:,1]
                mu_B, logvar_B = orig_age_encoding_B[:,0], orig_age_encoding_B[:,1]
                std_A = torch.exp(0.5 * logvar_A)
                std_B = torch.exp(0.5 * logvar_B)
                orig_age_features_A = std_A * torch.randn(b, self.cond_global_length).cuda() + mu_A
                orig_age_features_B = std_B * torch.randn(b, self.cond_global_length).cuda() + mu_B
            else:
                orig_age_features_A = orig_age_encoding_A
                orig_age_features_B = orig_age_encoding_B

            #within domain decode
            reconst_tex_images_A = self.netG_tex_A.decode(orig_id_features_A, None, orig_age_features_A)
            reconst_tex_images_B = self.netG_tex_B.decode(orig_id_features_B, None, orig_age_features_B)
            #cross domain decode
            generated_tex_images_BA = self.netG_tex_A.decode(orig_id_features_B, None, target_A)
            generated_tex_images_AB = self.netG_tex_B.decode(orig_id_features_A, None, target_B)
            #encode generated
            recon_id_features_B, recon_age_features_A = self.netG_tex_A.encode(generated_tex_images_BA, mask_B)
            recon_id_features_A, recon_age_features_B = self.netG_tex_B.encode(generated_tex_images_AB, mask_A)
            if self.vae_style_encoder:
                recon_mu_A, recon_logvar_A = recon_age_features_A[:,0], recon_age_features_A[:,1]
                recon_mu_B, recon_logvar_B = recon_age_features_B[:,0], recon_age_features_B[:,1]

            #decode generated
            cyc_tex_images_A = self.netG_tex_A.decode(recon_id_features_A, None, orig_age_features_A)
            cyc_tex_images_B = self.netG_tex_B.decode(recon_id_features_B, None, orig_age_features_B)
            #discriminator pass
            disc_out_A, _, _ = self.netD_tex_A(generated_tex_images_BA)
            disc_out_B, _, _ = self.netD_tex_B(generated_tex_images_AB)

            #self-reconstruction loss
            loss_G_ID_A = self.criterionID(reconst_tex_images_A, gen_in_A) * self.opt.lambda_id_tex
            loss_G_ID_B = self.criterionID(reconst_tex_images_B, gen_in_B) * self.opt.lambda_id_tex
            loss_G_ID = loss_G_ID_A + loss_G_ID_B
            #cycle loss
            loss_G_Cycle_A = self.criterionCycle(cyc_tex_images_A, gen_in_A) * self.opt.lambda_cyc_tex
            loss_G_Cycle_B = self.criterionCycle(cyc_tex_images_B, gen_in_B) * self.opt.lambda_cyc_tex
            loss_G_Cycle = loss_G_Cycle_A + loss_G_Cycle_B
            #content (identity) feature loss
            loss_G_content_reconst_A = self.content_reconst_criterion(recon_id_features_A, orig_id_features_A) * self.opt.lambda_content
            loss_G_content_reconst_B = self.content_reconst_criterion(recon_id_features_B, orig_id_features_B) * self.opt.lambda_content
            loss_G_content_reconst = loss_G_content_reconst_A + loss_G_content_reconst_B
            #age feature loss
            if self.vae_style_encoder:
                loss_G_age_reconst_A = self.age_embedding_criterion(recon_mu_A, recon_logvar_A) * self.opt.lambda_age
                loss_G_age_reconst_B = self.age_embedding_criterion(recon_mu_B, recon_logvar_B) * self.opt.lambda_age
            else:
                loss_G_age_reconst_A = self.age_reconst_criterion(recon_age_features_A, target_A) * self.opt.lambda_age
                loss_G_age_reconst_B = self.age_reconst_criterion(recon_age_features_B, target_B) * self.opt.lambda_age

            loss_G_age_reconst = loss_G_age_reconst_A + loss_G_age_reconst_B
            #age feature embedding loss
            # loss_G_age_embedding = self.age_embedding_criterion(orig_age_features, self.orig_age_mean, self.orig_age_sigma) * self.opt.lambda_embedding
            if self.embed_latent:
                loss_G_age_embedding_A = self.age_embedding_criterion(target_A, orig_age_features_A, orig_age_features_B) * self.opt.lambda_embedding
                loss_G_age_embedding_B = self.age_embedding_criterion(target_B, orig_age_features_B, orig_age_features_A) * self.opt.lambda_embedding
                loss_G_age_embedding = loss_G_age_embedding_A + loss_G_age_embedding_B
            elif self.vae_style_encoder:
                loss_G_age_embedding_A = self.age_embedding_criterion(mu_A, logvar_A)
                loss_G_age_embedding_B = self.age_embedding_criterion(mu_B, logvar_B)
                loss_G_age_embedding = loss_G_age_embedding_A + loss_G_age_embedding_B
            else:
                loss_G_age_embedding = torch.zeros(1).cuda()

            #GAN loss
            # target_classes = torch.cat((self.class_B,self.class_A),0)
            loss_G_GAN_A = self.criterionGAN(disc_out_A, True)
            loss_G_GAN_B = self.criterionGAN(disc_out_B, True)
            loss_G_GAN = loss_G_GAN_A + loss_G_GAN_B
        else:
            self.get_conditions()
            if self.gan_mode == 'seg_and_texture':
                #prepare segmentation generator inputs
                if self.use_expanded_parsings:
                    seg_gen_in = self.expanded_parsings
                else:
                    # seg_gen_in = self.parsings
                    seg_gen_in = util.flip_eye_labels(self.parsings)

                # remove clothing items from texture input
                if self.no_background and not self.mask_gen_outputs:
                    tex_gen_in = self.masked_reals
                else:
                    tex_gen_in = self.reals

                if self.no_clothing_items:
                    tex_gen_in = tex_gen_in * (1-self.clothing_items_masks)
                    self.masks = self.masks * (1-self.clothing_items_masks)

                if self.no_neck_tex:
                    tex_gen_in = tex_gen_in * (1-self.neck_masks)
                    self.masks = self.masks * (1-self.neck_masks)

                #embed gaussians into target style space
                if self.embed_latent:
                    gen_embeddings = self.target_embedding(self.gen_conditions)
                    rec_embeddings = self.target_embedding(self.rec_conditions)
                else:
                    gen_embeddings = self.gen_conditions
                    rec_embeddings = self.rec_conditions

                ################## segmentation forward pass ###################
                seg_grad_status = torch.no_grad() if self.seg_fixed else torch.enable_grad()
                with seg_grad_status:
                    # within domain pass
                    reconst_seg_image, orig_seg_id_features, orig_seg_age_features = self.netG_seg(seg_gen_in, None, rec_embeddings)
                    # cross domain pass
                    generated_seg_images, _, _ = self.netG_seg(gen_in, None, gen_embeddings)
                    # cycle pass
                    cyc_seg_images, fake_seg_id_features, fake_seg_age_features = self.netG_seg(generated_seg_images, None, rec_embeddings)

                    # #encode original segmentations
                    # orig_seg_id_features, orig_seg_age_features = self.netG_seg.module.encode(seg_gen_in)
                    # #within domain decode
                    # if self.use_orig_age_features_within_domain:
                    #     reconst_seg_images = self.netG_seg(seg_gen_in, None)
                    # else:
                    #     reconst_seg_images = self.netG_seg.module.decode(orig_seg_id_features, None, rec_embeddings)
                    # #cross domain decode
                    # generated_seg_images = self.netG_seg.module.decode(orig_seg_id_features, None, gen_embeddings)
                    # #encode generated
                    # fake_seg_id_features, fake_seg_age_features = self.netG_seg.module.encode(generated_seg_images)
                    # #decode generated
                    # cyc_seg_images = self.netG_seg.module.decode(fake_seg_id_features, None, rec_embeddings)

                ################## segmentation losses ###################
                if self.seg_fixed:
                    loss_G_ID_seg = 0
                    loss_G_Cycle_seg = 0
                    loss_G_content_reconst_seg = 0
                    loss_G_age_reconst_seg = 0
                    loss_G_age_embedding_seg = torch.zeros(1).cuda()
                else:
                    #self-reconstruction loss
                    loss_G_ID_seg = self.criterionID(reconst_seg_images, seg_gen_in) * self.opt.lambda_id_tex
                    #cycle loss
                    loss_G_Cycle_seg = self.criterionCycle(cyc_seg_images, seg_gen_in) * self.opt.lambda_cyc_tex
                    #content (identity) feature loss
                    loss_G_content_reconst_seg = self.content_reconst_criterion(fake_seg_id_features, orig_seg_id_features) * self.opt.lambda_content
                    #age feature loss
                    loss_G_age_reconst_seg = self.age_reconst_criterion(fake_seg_age_features, gen_embeddings) * self.opt.lambda_age
                    #age feature embedding loss
                    # loss_G_age_embedding_seg = self.age_embedding_criterion(orig_seg_age_features, self.orig_seg_age_mean, self.orig_seg_age_sigma) * self.opt.lambda_embedding
                    if self.embed_latent:
                        b = orig_seg_age_features.shape[0]
                        flipped_seg_age_embeddings = torch.cat((orig_seg_age_features[b:,:,:,:],orig_seg_age_features[:b,:,:,:]),0)
                        loss_G_age_embedding_seg = self.age_embedding_criterion(gen_embeddings, flipped_seg_age_embeddings, orig_seg_age_features) * self.opt.lambda_embedding
                    else:
                        loss_G_age_embedding_seg = torch.zeros(1).cuda()

                ################## texture forward pass ###################
                # within domain pass
                reconst_tex_image, orig_tex_id_features, orig_tex_age_features = self.netG_tex(tex_gen_in, seg_gen_in, rec_embeddings)
                # cross domain pass
                generated_tex_images, _, _ = self.netG_tex(gen_in, generated_seg_image, gen_embeddings)
                # cycle pass
                cyc_tex_images, fake_tex_id_features, fake_tex_age_features = self.netG_tex(generated_tex_images, seg_gen_in, rec_embeddings)

                # # #encode original images
                # orig_tex_id_features, orig_tex_age_features = self.netG_tex.module.encode(tex_gen_in)
                # #within domain decode
                # if self.use_orig_age_features_within_domain:
                #     reconst_tex_images = self.netG_tex(tex_gen_in, seg_gen_in)
                # else:
                #     reconst_tex_images = self.netG_tex.module.decode(orig_tex_id_features, seg_gen_in, rec_embeddings)
                #
                # #cross domain decode
                # generated_tex_images = self.netG_tex.module.decode(orig_tex_id_features, generated_seg_images, gen_embeddings)
                # #encode generated
                # fake_tex_id_features, fake_tex_age_features = self.netG_tex.module.encode(generated_tex_images)
                # #decode generated
                # cyc_tex_images = self.netG_tex.module.decode(fake_tex_id_features, seg_gen_in, rec_embeddings)

                #discriminator pass
                if self.use_parsings_in_disc:
                    disc_input = torch.cat((generated_tex_images, generated_seg_images), 1)
                else:
                    disc_input = generated_tex_images

                if self.add_disc_cond_tex_class:
                    disc_input = torch.cat((disc_input, self.fake_disc_conditions),1)

                disc_out, _, _ = self.netD_tex(disc_input)

                ################## texture losses ###################
                #self-reconstruction loss
                loss_G_ID_tex = self.criterionID(reconst_tex_images, tex_gen_in) * self.opt.lambda_id_tex
                #cycle loss
                loss_G_Cycle_tex = self.criterionCycle(cyc_tex_images, tex_gen_in) * self.opt.lambda_cyc_tex
                #content (identity) feature loss
                loss_G_content_reconst_tex = self.content_reconst_criterion(fake_tex_id_features, orig_tex_id_features) * self.opt.lambda_content
                #age feature loss
                loss_G_age_reconst_tex = self.age_reconst_criterion(fake_tex_age_features, gen_embeddings) * self.opt.lambda_age
                #age feature embedding loss
                # loss_G_age_embedding = self.age_embedding_criterion(orig_age_features, self.orig_age_mean, self.orig_age_sigma) * self.opt.lambda_embedding
                if self.embed_latent:
                    b = orig_age_features.shape[0]
                    flipped_tex_age_embeddings = torch.cat((orig_tex_age_features[b:,:,:,:],orig_tex_age_features[:b,:,:,:]),0)
                    loss_G_age_embedding_tex = self.age_embedding_criterion(gen_embeddings, flipped_tex_age_embeddings, orig_tex_age_features) * self.opt.lambda_embedding
                else:
                    loss_G_age_embedding_tex = torch.zeros(1).cuda()

                loss_G_ID = loss_G_ID_seg + loss_G_ID_tex
                loss_G_Cycle = loss_G_Cycle_seg + loss_G_Cycle_tex
                loss_G_content_reconst = loss_G_content_reconst_seg + loss_G_content_reconst_seg
                loss_G_age_reconst = loss_G_age_reconst_seg + loss_G_age_reconst_tex
                loss_G_age_embedding = loss_G_age_embedding_seg + loss_G_age_embedding_tex

                # #GAN loss
                # target_classes = torch.cat((self.class_B,self.class_A),0)
                # loss_G_GAN = self.criterionGAN(disc_out, target_classes, True, is_gen=True)
            else:
                if self.gan_mode == 'seg_only':
                    if self.use_expanded_parsings:
                        gen_in = self.expanded_parsings
                    else:
                        gen_in = self.parsings

                    generator = getattr(self, 'netG_seg')
                    discriminator = getattr(self, 'netD_seg')
                    if self.use_moving_avg:
                        g_running = getattr(self, 'g_running_seg')
                if self.gan_mode == 'texture_only':
                    # remove clothing items from texture input
                    if self.no_background and not self.mask_gen_outputs:
                        gen_in = self.masked_reals
                    else:
                        gen_in = self.reals

                    if self.no_clothing_items:
                        gen_in = gen_in * (1-self.clothing_items_masks)
                        self.masks = self.masks * (1-self.clothing_items_masks)
                        self.parsings = util.removeClothingItems(self.parsings, self.clothing_items_masks)

                    if self.no_neck_tex:
                        gen_in = gen_in * (1-self.neck_masks)
                        self.masks = self.masks * (1-self.neck_masks)
                        self.parsings = util.removeNeck(self.parsings, self.neck_masks)

                    generator = getattr(self, 'netG_tex')
                    discriminator = getattr(self, 'netD_tex')
                    if self.use_moving_avg:
                        g_running = getattr(self, 'g_running')

                #embed gaussians into target style space
                if self.embed_latent:
                    gen_embeddings = self.target_embedding(self.gen_conditions)
                    rec_embeddings = self.target_embedding(self.rec_conditions)
                    orig_embeddings = self.target_embedding(self.orig_conditions)
                else:
                    gen_embeddings = self.gen_conditions
                    rec_embeddings = self.rec_conditions
                    orig_embeddings = self.orig_conditions

                # # within domain pass
                # if self.opt.lambda_id_tex > 0:
                #     # reconst_tex_images, _, _ = generator(gen_in, None, rec_embeddings)
                #     reconst_tex_images, orig_id_features, orig_age_features = generator(gen_in, None, rec_embeddings)
                #
                # # cross domain pass
                # # generated_tex_images, orig_id_features, orig_age_features = generator(gen_in, None, gen_embeddings)
                # generated_tex_images, _, _ = generator(gen_in, None, gen_embeddings)
                # # cycle pass
                # cyc_tex_images, fake_id_features, fake_age_features = generator(generated_tex_images, None, rec_embeddings)
                ############### multi GPU ###############
                reconst_tex_images, generated_tex_images, cyc_tex_images, orig_id_features, orig_age_features, \
                fake_id_features, fake_age_features, gen_flow, gen_seg, rec_xy, cyc_xy = \
                generator(gen_in, rec_embeddings, gen_embeddings, orig_embeddings, xy=self.xy, seg=self.parsings)

                ############### single GPU ###############
                # #encode images
                # orig_id_features, orig_age_features = generator.module.encode(gen_in)
                # #within domain decode
                # if self.use_orig_age_features_within_domain:
                #     reconst_tex_images = generator(gen_in, None)
                # else:
                #     reconst_tex_images = generator.module.decode(orig_id_features, None, rec_embeddings)
                # #cross domain decode
                # generated_tex_images = generator.module.decode(orig_id_features, None, gen_embeddings)
                #
                # #encode generated
                # fake_id_features, fake_age_features = generator.module.encode(generated_tex_images, self.masks)
                # #decode generated
                # if self.opt.lambda_cyc_tex > 0:
                #     cyc_tex_images = generator.module.decode(fake_id_features, None, rec_embeddings)

                #discriminator pass
                if self.gan_mode == 'seg_only' and self.use_expanded_parsings:
                    disc_in = util.probs2rgb(generated_tex_images)
                else:
                    if self.use_flow_layers:
                        if self.two_discriminators:
                            disc_in = generated_tex_images
                            seg_disc_in = gen_seg
                        else:
                            disc_in = torch.cat((generated_tex_images, gen_seg), 1)
                    else:
                        disc_in = generated_tex_images

                if self.add_disc_cond_tex_class:
                    disc_in = torch.cat((disc_in, self.fake_disc_conditions),1)

                disc_out, _, _ = discriminator(disc_in)
                if self.use_flow_layers and self.two_discriminators:
                    seg_disc_out, _, _ = self.netD_seg(seg_disc_in)

                #self-reconstruction loss
                if self.opt.lambda_id_tex > 0:
                    loss_G_ID = self.criterionID(reconst_tex_images, gen_in) * self.opt.lambda_id_tex
                else:
                    loss_G_ID = torch.zeros(1).cuda()

                #flow self-reconstruction loss
                if self.use_flow_layers and self.opt.lambda_id_flow > 0:
                    loss_G_ID_flow = self.criterionID(rec_xy, self.xy) * self.opt.lambda_id_flow
                else:
                    loss_G_ID_flow = torch.zeros(1).cuda()

                #background consistency loss
                if self.use_background_loss:
                    background_masks = 1- self.masks
                    masked_generated = background_masks * generated_tex_images
                    masked_target = background_masks * gen_in
                    bkgnd_correction = background_masks.numel() / background_masks.sum()
                    Loss_G_background = self.criterionID(masked_generated, masked_target) * bkgnd_correction * self.opt.lambda_id_tex
                else:
                    Loss_G_background = 0

                #cycle loss
                if self.opt.lambda_cyc_tex > 0:
                    loss_G_Cycle = self.criterionCycle(cyc_tex_images, gen_in) * self.opt.lambda_cyc_tex
                else:
                    loss_G_Cycle = torch.zeros(1).cuda()

                #flow cycle loss
                if self.use_flow_layers and self.opt.lambda_cyc_flow > 0:
                    loss_G_Cycle_flow = self.criterionCycle(cyc_xy, self.xy) * self.opt.lambda_cyc_flow
                else:
                    loss_G_Cycle_flow = torch.zeros(1).cuda()

                #flow TV loss
                if self.use_flow_layers and self.opt.lambda_flowTV > 0:
                    loss_flowTV = self.criterionFlowTV(gen_flow) * self.opt.lambda_flowTV
                else:
                    loss_flowTV = torch.zeros(1).cuda()

                #content (identity) feature loss
                loss_G_content_reconst = self.content_reconst_criterion(fake_id_features, orig_id_features) * self.opt.lambda_content
                #age feature loss
                loss_G_age_reconst = self.age_reconst_criterion(fake_age_features, gen_embeddings) * self.opt.lambda_age
                #orig age feature loss
                if self.orig_age_features_rec_penalty:
                    loss_G_age_reconst += self.age_reconst_criterion(orig_age_features, orig_embeddings) * self.opt.lambda_age

                #age feature embedding loss
                # loss_G_age_embedding = self.age_embedding_criterion(orig_age_features, self.orig_age_mean, self.orig_age_sigma) * self.opt.lambda_embedding
                if self.embed_latent:
                    b = orig_age_features.shape[0]
                    flipped_age_embeddings = torch.cat((orig_age_features[b:,:,:,:],orig_age_features[:b,:,:,:]),0)
                    loss_G_age_embedding = self.age_embedding_criterion(gen_embeddings, flipped_age_embeddings, orig_age_features) * self.opt.lambda_embedding
                else:
                    loss_G_age_embedding = torch.zeros(1).cuda()

            #GAN loss
            target_classes = torch.cat((self.class_B,self.class_A),0)
            loss_G_GAN = self.criterionGAN(disc_out, target_classes, True, is_gen=True)
            if self.use_flow_layers and self.two_discriminators:
                loss_G_GAN += self.criterionGAN(seg_disc_out, target_classes, True, is_gen=True)

        loss_G = (loss_G_GAN + loss_G_ID + Loss_G_background + loss_G_Cycle + loss_G_content_reconst + \
                  loss_G_age_reconst + loss_G_age_embedding + loss_G_ID_flow + loss_G_Cycle_flow + loss_flowTV).mean()

        loss_G.backward()
        self.optimizer_G.step()

        if self.use_moving_avg:
            self.accumulate(g_running, generator)

        if infer:
            if self.original_munit:
                generated_tex_images_out = torch.cat((generated_tex_images_AB, generated_tex_images_BA), 0)
                if self.opt.lambda_id_tex > 0:
                    reconst_tex_images_out = torch.cat((reconst_tex_images_A, reconst_tex_images_B), 0)
                if self.opt.lambda_cyc_tex > 0:
                    cyc_tex_images_out = torch.cat((cyc_tex_images_A, cyc_tex_images_B), 0)
            elif self.use_moving_avg:
                with torch.no_grad():
                    orig_id_features_out, _ = g_running.encode(gen_in)
                    #within domain decode
                    if self.opt.lambda_id_tex > 0:
                        if self.use_orig_age_features_within_domain:
                            reconst_tex_images_out = g_running(gen_in, None)
                        else:
                            reconst_tex_images_out, _, _, reconst_seg_images = g_running.decode(orig_id_features_out, None, rec_embeddings, flow_seg=self.parsings)

                    #cross domain decode
                    generated_tex_images_out, _, _, generated_seg_images = g_running.decode(orig_id_features_out, None, gen_embeddings, flow_seg=self.parsings)
                    #encode generated
                    fake_id_features_out, _ = g_running.encode(generated_tex_images, self.masks)
                    #decode generated
                    if self.opt.lambda_cyc_tex > 0:
                        cyc_tex_images_out, _, _, cyc_seg_images = g_running.decode(fake_id_features_out, None, rec_embeddings, flow_seg=generated_seg_images)
                        if self.gan_mode == 'seg_only' and self.use_expanded_parsings:
                            cyc_tex_images_out = util.probs2rgb(cyc_tex_images_out)

                    if self.gan_mode == 'seg_only' and self.use_expanded_parsings:
                        reconst_tex_images_out = util.probs2rgb(reconst_tex_images_out)
                        generated_tex_images_out = util.probs2rgb(generated_tex_images_out)
            else:
                generated_tex_images_out = generated_tex_images
                if self.opt.lambda_id_tex > 0:
                    reconst_tex_images_out = reconst_tex_images
                if self.opt.lambda_cyc_tex > 0:
                    cyc_tex_images_out = cyc_tex_images

        loss_dict = {'loss_G_GAN': loss_G_GAN.mean(), 'loss_G_Cycle': loss_G_Cycle.mean(), 'loss_G_ID': loss_G_ID.mean(),
                     'loss_G_content_reconst': loss_G_content_reconst.mean(), 'loss_G_age_reconst': loss_G_age_reconst.mean(),
                     'loss_G_age_embedding': loss_G_age_embedding.mean()}
        if self.use_flow_layers:
            loss_dict.update({'loss_G_ID_flow': loss_G_ID_flow.mean(), 'loss_G_Cycle_flow': loss_G_Cycle_flow.mean(),
                              'loss_flowTV': loss_flowTV.mean()})
        if self.use_background_loss:
            loss_dict.update({'loss_G_background': Loss_G_background})

        return [loss_dict,
                None if not infer else gen_in,
                None if not (infer and 'flow' in self.gan_mode) else warped_in[:,0:3,:,:],
                None if not (infer and 'flow' in self.gan_mode) else fake_parsings,
                None if not (infer and 'flow' in self.gan_mode) else fake_grid,
                None if not (infer and ('texture' in self.gan_mode or 'seg' in self.gan_mode)) else generated_tex_images_out,
                None if not (infer and ('texture' in self.gan_mode or 'seg' in self.gan_mode) and (self.opt.lambda_id_tex > 0)) else reconst_tex_images_out,
                None if not (infer and ('texture' in self.gan_mode or 'seg' in self.gan_mode) and (self.opt.lambda_cyc_tex > 0)) else cyc_tex_images_out,
                None if not (infer and (self.use_flow_layers or self.gan_mode == 'seg_and_texture')) else generated_seg_images,
                None if not (infer and (self.use_flow_layers or self.gan_mode == 'seg_and_texture')) else reconst_seg_images,
                None if not (infer and (self.use_flow_layers or self.gan_mode == 'seg_and_texture')) else cyc_seg_images]

    def update_D(self):
        self.optimizer_D.zero_grad()
        if self.original_munit:
            # remove clothing items from texture input
            if self.no_background and not self.mask_gen_outputs:
                gen_in = self.masked_reals
            else:
                gen_in = self.reals

            if self.no_clothing_items:
                gen_in = gen_in * (1-self.clothing_items_masks)
                self.masks = self.masks * (1-self.clothing_items_masks)

            if self.no_neck_tex:
                gen_in = gen_in * (1-self.neck_masks)
                self.masks = self.masks * (1-self.neck_masks)

            b = self.class_A.shape[0]
            if self.class_A == 0:
                gen_in_A = gen_in[:b,:,:,:]
                gen_in_B = gen_in[b:,:,:,:]
                mask_A = self.masks[:b,:,:,:]
                mask_B = self.masks[b:,:,:,:]
            else:
                gen_in_A = gen_in[b:,:,:,:]
                gen_in_B = gen_in[:b,:,:,:]
                mask_A = self.masks[b:,:,:,:]
                mask_B = self.masks[:b,:,:,:]

            # generate random gaussians
            target_A = torch.randn(b, self.cond_global_length).cuda()
            target_B = torch.randn(b, self.cond_global_length).cuda()
            #encode images
            orig_id_features_A, orig_age_features_A = self.netG_tex_A.encode(gen_in_A, mask_A)
            orig_id_features_B, orig_age_features_B = self.netG_tex_B.encode(gen_in_B, mask_B)
            #cross domain decode
            generated_tex_images_BA = self.netG_tex_A.decode(orig_id_features_B, None, target_A)
            generated_tex_images_AB = self.netG_tex_B.decode(orig_id_features_A, None, target_B)
            #fake discriminator pass
            fake_disc_out_A, _, _ = self.netD_tex_A(generated_tex_images_BA.detach())
            fake_disc_out_B, _, _ = self.netD_tex_B(generated_tex_images_AB.detach())
            #real discriminator pass
            real_disc_out_A, _, _ = self.netD_tex_A(gen_in_A)
            real_disc_out_B, _, _ = self.netD_tex_B(gen_in_B)

            #Fake GAN loss
            loss_D_fake_A = self.criterionGAN(fake_disc_out_A, False)
            loss_D_fake_B = self.criterionGAN(fake_disc_out_B, False)
            loss_D_fake = loss_D_fake_A + loss_D_fake_B

            #Real GAN loss
            loss_D_real_A = self.criterionGAN(real_disc_out_A, True)
            loss_D_real_B = self.criterionGAN(real_disc_out_B, True)
            loss_D_real = loss_D_real_A + loss_D_real_B

        else:
            self.get_conditions()

            if self.gan_mode == 'seg_and_texture':
                if self.use_expanded_parsings:
                    seg_gen_in = self.expanded_parsings
                else:
                    # seg_gen_in = self.parsings
                    seg_gen_in = util.flip_eye_labels(self.parsings)

                # remove clothing items from texture input
                if self.no_background and not self.mask_gen_outputs:
                    tex_gen_in = self.masked_reals
                else:
                    tex_gen_in = self.reals

                if self.no_clothing_items:
                    tex_gen_in = tex_gen_in * (1-self.clothing_items_masks)
                    self.masks = self.masks * (1-self.clothing_items_masks)

                if self.no_neck_tex:
                    tex_gen_in = tex_gen_in * (1-self.neck_masks)
                    self.masks = self.masks * (1-self.neck_masks)

                # cross domain pass
                generated_seg_images, _, _ = self.netG_seg(gen_in, None, self.gen_conditions)
                generated_tex_images, _, _ = self.netG_tex(orig_tex_id_features, generated_seg_images, self.gen_conditions)

                # #encode segmentations & images
                # orig_seg_id_features, orig_seg_age_features = self.netG_seg.module.encode(seg_gen_in)
                # orig_tex_id_features, orig_tex_age_features = self.netG_tex.module.encode(tex_gen_in)
                # #cross domain decode
                # generated_seg_images = self.netG_seg.module.decode(orig_seg_id_features, None, self.gen_conditions)
                # generated_tex_images = self.netG_tex.module.decode(orig_tex_id_features, generated_seg_images, self.gen_conditions)

                #fake discriminator pass
                if self.use_parsings_in_disc:
                    fake_disc_in = torch.cat((generated_tex_images.detach(), generated_seg_images.detach()), 1)
                else:
                    fake_disc_in = generated_tex_images

                if self.add_disc_cond_tex_class:
                    fake_disc_in = torch.cat((fake_disc_in, self.fake_disc_conditions),1)

                fake_disc_out, _, _ = self.netD_tex(fake_disc_in)
                #real discriminator pass
                if self.use_parsings_in_disc:
                    real_disc_in = torch.cat((tex_gen_in, seg_gen_in), 1)
                else:
                    real_disc_in = tex_gen_in

                if self.add_disc_cond_tex_class:
                    real_disc_in = torch.cat((real_disc_in, self.real_disc_conditions),1)

                if (self.selective_class_type_tex == 'hinge') or (self.selective_class_type_tex == 'non_sat'):
                    # necessary for R1 regularization
                    real_disc_in.requires_grad_()

                real_disc_out, _, _ = self.netD_tex(real_disc_in)

                #Fake GAN loss
                fake_target_classes = torch.cat((self.class_B,self.class_A),0)
                loss_D_fake = self.criterionGAN(fake_disc_out, fake_target_classes, False, is_gen=False)

                #Real GAN loss
                real_target_classes = torch.cat((self.class_A,self.class_B),0)
                loss_D_real = self.criterionGAN(real_disc_out, real_target_classes, True, is_gen=False)

                #R1 regularization (when necessary)
                if (self.selective_class_type_tex == 'hinge') or (self.selective_class_type_tex == 'non_sat'):
                    loss_D_reg = self.R1_reg(real_disc_out, real_disc_in)
                else:
                    loss_D_reg = 0
            else:
                if self.gan_mode == 'seg_only':
                    if self.use_expanded_parsings:
                        gen_in = self.expanded_parsings
                    else:
                        gen_in = self.parsings

                    generator = getattr(self, 'netG_seg')
                    discriminator = getattr(self, 'netD_seg')
                if self.gan_mode == 'texture_only':
                    # remove clothing items from texture input
                    if self.no_background and not self.mask_gen_outputs:
                        gen_in = self.masked_reals
                    else:
                        gen_in = self.reals

                    if self.no_clothing_items:
                        gen_in = gen_in * (1-self.clothing_items_masks)
                        self.masks = self.masks * (1-self.clothing_items_masks)
                        self.parsings = util.removeClothingItems(self.parsings, self.clothing_items_masks)

                    if self.no_neck_tex:
                        gen_in = gen_in * (1-self.neck_masks)
                        self.masks = self.masks * (1-self.neck_masks)
                        self.parsings = util.removeNeck(self.parsings, self.neck_masks)

                    generator = getattr(self, 'netG_tex')
                    discriminator = getattr(self, 'netD_tex')

                # cross domain pass
                # generated_tex_images, _, _ = generator(gen_in, None, self.gen_conditions)
                ############### multi GPU ###############
                _, generated_tex_images, _, _, _, _, _, _, generated_seg_images, _, _ = generator(gen_in, None, self.gen_conditions, None, disc_pass=True, seg=self.parsings)

                ############### single GPU ###############
                # #encode images
                # orig_id_features, orig_age_features = generator.module.encode(gen_in)
                # #cross domain decode
                # generated_tex_images = generator.module.decode(orig_id_features, None, self.gen_conditions)

                #fake discriminator pass
                #discriminator pass
                if self.gan_mode == 'seg_only' and self.use_expanded_parsings:
                    fake_disc_in = util.probs2rgb(generated_tex_images).detach()
                else:
                    if self.use_flow_layers:
                        if self.two_discriminators:
                            fake_disc_in = generated_tex_images.detach()
                            fake_seg_disc_in = generated_seg_images.detach()
                        else:
                            fake_disc_in = torch.cat((generated_tex_images.detach(), generated_seg_images.detach()), 1)
                    else:
                        fake_disc_in = generated_tex_images.detach()

                if self.add_disc_cond_tex_class:
                    fake_disc_in = torch.cat((fake_disc_in, self.fake_disc_conditions),1)

                fake_disc_out, _, _ = discriminator(fake_disc_in)
                if self.use_flow_layers and self.two_discriminators:
                    fake_seg_disc_out, _, _ = self.netD_seg(fake_seg_disc_in)

                #real discriminator pass
                if self.gan_mode == 'seg_only' and self.use_expanded_parsings:
                    real_disc_in = util.probs2rgb(gen_in)
                else:
                    if self.use_flow_layers:
                        if self.two_discriminators:
                            real_disc_in = gen_in
                            real_seg_disc_in = self.parsings
                        else:
                            real_disc_in = torch.cat((gen_in, self.parsings), 1)
                    else:
                        real_disc_in = gen_in

                if self.add_disc_cond_tex_class:
                    real_disc_in = torch.cat((real_disc_in, self.real_disc_conditions),1)

                if (self.selective_class_type_tex == 'hinge') or (self.selective_class_type_tex == 'non_sat'):
                    # necessary for R1 regularization
                    real_disc_in.requires_grad_()
                    if self.use_flow_layers and self.two_discriminators:
                        real_seg_disc_in.requires_grad_()

                real_disc_out, _, _ = discriminator(real_disc_in)
                if self.use_flow_layers and self.two_discriminators:
                    real_seg_disc_out, _, _ = self.netD_seg(real_seg_disc_in)

                #Fake GAN loss
                fake_target_classes = torch.cat((self.class_B,self.class_A),0)
                loss_D_fake = self.criterionGAN(fake_disc_out, fake_target_classes, False, is_gen=False)
                if self.use_flow_layers and self.two_discriminators:
                    loss_D_fake += self.criterionGAN(fake_seg_disc_out, fake_target_classes, False, is_gen=False)

                #Real GAN loss
                real_target_classes = torch.cat((self.class_A,self.class_B),0)
                loss_D_real = self.criterionGAN(real_disc_out, real_target_classes, True, is_gen=False)
                if self.use_flow_layers and self.two_discriminators:
                    loss_D_real += self.criterionGAN(real_seg_disc_out, real_target_classes, True, is_gen=False)

                # R1 regularization (when necessary)
                if (self.selective_class_type_tex == 'hinge') or (self.selective_class_type_tex == 'non_sat'):
                    loss_D_reg = self.R1_reg(real_disc_out, real_disc_in)
                    if self.use_flow_layers and self.two_discriminators:
                        loss_D_reg += self.R1_reg(real_seg_disc_out, real_seg_disc_in)
                else:
                    loss_D_reg = torch.zeros(1).cuda()

        loss_D = (loss_D_fake + loss_D_real + loss_D_reg).mean()
        loss_D.backward()
        self.optimizer_D.step()

        return {'loss_D_real': loss_D_real.mean(), 'loss_D_fake': loss_D_fake.mean(), 'loss_D_reg': loss_D_reg.mean()}

    def forward(self, data, infer=False, optG_wgan=False):
        # Encode Inputs
        self.encode_input(data)
        self.get_conditions()

        if self.gan_mode == 'flow_only':
            if self.use_encoding_net_flow:
                labels = None

                flow_enc_input = torch.cat((self.parsings,self.landmarks), 1)
                flow_enc_cond = self.downsampler(self.flow_gen_conditions)
                flow_features = self.netE_flow(flow_enc_input, flow_enc_cond, labels)
            else:
                flow_features = None

            if self.no_background and not self.mask_gen_outputs:
                rgb_in = self.masked_reals
            else:
                rgb_in = self.reals

            fake_warped_images, fake_masked_warped_images, fake_grid, fake_mask, \
            fake_parsings, _, _, _, _, fake_landmarks, warped_xy = \
            self.netG_flow(rgb_in, self.flow_gen_conditions, parsing=self.parsings,
                                   mask=self.masks, landmarks=self.landmarks, features=flow_features,
                                   expanded_parsings=self.expanded_parsings, xy=self.xy,
                                   use_rgbxy_inputs=self.use_rgbxy_flow_inputs,
                                   use_xy_inputs=self.use_xy_flow_inputs)
            if self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                fake_flow_images = torch.cat((fake_parsings, fake_landmarks),1)
            elif self.use_parsings or (self.use_landmarks and self.json_landmarks):
                fake_flow_images = fake_parsings
            elif self.use_landmarks and (not self.json_landmarks):
                fake_flow_images = fake_landmarks
            else:
                fake_flow_images = fake_masked_warped_images

            fake_tex_images = None
            fake_images = (fake_flow_images, None)

        if self.gan_mode == 'texture_only':
            # remove clothing items from texture input
            tex_parsings = self.parsings.detach()
            if self.no_background and not self.mask_gen_outputs:
                rgb_in = self.masked_reals
            else:
                rgb_in = self.reals

            if self.no_clothing_items:
                rgb_in = rgb_in * (1-self.clothing_items_masks)
                tex_parsings = util.removeClothingItems(tex_parsings, self.clothing_items_masks)

            if self.no_neck_tex:
                rgb_in = rgb_in * (1-self.neck_masks)
                tex_parsings = util.removeNeck(tex_parsings, self.neck_masks)

            # add parsings to texture input
            if self.use_encoding_net:
                if self.use_avg_features:
                    labels = self.expanded_parsings
                else:
                    labels = None

                # we need to downsample the conditions once more
                # since the encoder has 4 downsampling layers while the generator has 3
                if self.downsample_tex:
                    enc_cond = self.downsampler(self.gen_conditions)
                else:
                    enc_cond = self.downsampler(self.gen_conditions[0])

                features = self.netE_tex(self.masked_real, enc_cond, labels)
                if self.use_parsings_tex_in:
                    tex_in = torch.cat((features, self.expanded_parsings), 1)
                    init_tex_in = torch.cat((rgb_in, tex_parsings), 1)
                else:
                    tex_in = features
                    init_tex_in = rgb_in

            elif self.use_parsings_tex_in:
                # if self.no_facial_hair:
                #     tex_parsings = util.restoreFacialHair(tex_parsings, self.facial_hair_masks)
                tex_in = torch.cat((rgb_in, tex_parsings),1)
            else:
                tex_in = rgb_in

            # forward pass of texture net
            if 'ada' in self.netG_tex_arch:
                tex_in_id_features, tex_in_age_features = self.netG_tex.encode(tex_in, self.masks)
                generated_tex_images = self.netG_tex.decode(tex_in_id_features, tex_in_age_features, self.gen_conditions)
            else:
                generated_tex_images = self.netG_tex(tex_in, self.gen_conditions)

            if self.use_parsings_in_disc and not self.use_parsings_tex_out:
                fake_tex_images = torch.cat((generated_tex_images, tex_parsings), 1)
            else:
                fake_tex_images = generated_tex_images

            fake_flow_images = None
            fake_images = (None, fake_tex_images)

        if self.gan_mode == 'flow_and_texture':
            if self.no_background and not self.mask_gen_outputs:
                rgb_in = self.masked_reals
            else:
                rgb_in = self.reals

            if self.flow_fixed:
                with torch.no_grad():
                    if self.use_encoding_net_flow:
                        labels = None

                        flow_enc_input = torch.cat((self.parsings,self.landmarks), 1)
                        flow_enc_cond = self.downsampler(self.flow_gen_conditions)
                        flow_features = self.netE_flow(flow_enc_input, flow_enc_cond, labels)
                    else:
                        flow_features = None

                    fake_warped_images, fake_masked_warped_images, fake_grid, fake_mask, \
                    fake_parsings, warped_facial_hair_mask, warped_clothing_items_mask, \
                    warped_neck_mask, warped_expanded_parsings, fake_landmarks, warped_xy = \
                    self.netG_flow(rgb_in, self.flow_gen_conditions, parsing=self.parsings,
                                   mask=self.masks, facial_hair_mask=self.facial_hair_masks,
                                   clothing_items_mask=self.clothing_items_masks, neck_mask=self.neck_masks,
                                   expanded_parsings=self.expanded_parsings, landmarks=self.landmarks,
                                   features=flow_features, xy=self.xy, use_rgbxy_inputs=self.use_rgbxy_flow_inputs,
                                   use_xy_inputs=self.use_xy_flow_inputs)
            else:
                if self.use_encoding_net_flow:
                    labels = None

                    flow_enc_input = torch.cat((self.parsings,self.landmarks), 1)
                    flow_enc_cond = self.downsampler(self.flow_gen_conditions)
                    flow_features = self.netE_flow(flow_enc_input, flow_enc_cond, labels)
                else:
                    flow_features = None

                fake_warped_images, fake_masked_warped_images, fake_grid, fake_mask, \
                fake_parsings, warped_facial_hair_mask, warped_clothing_items_mask, \
                warped_neck_mask, warped_expanded_parsings, fake_landmarks, warped_xy = \
                self.netG_flow(rgb_in, self.flow_gen_conditions, parsing=self.parsings,
                               mask=self.masks, facial_hair_mask=self.facial_hair_masks,
                               clothing_items_mask=self.clothing_items_masks, neck_mask=self.neck_masks,
                               expanded_parsings=self.expanded_parsings, landmarks=self.landmarks,
                               features=flow_features, xy=self.xy, use_rgbxy_inputs=self.use_rgbxy_flow_inputs,
                               use_xy_inputs=self.use_xy_flow_inputs)

            fake_warped_images_tex_in = fake_warped_images#.detach()
            tex_parsings = fake_parsings#.detach()
            # remove clothing items from texture input
            if self.no_background and not self.mask_gen_outputs:
                if self.no_clothing_items:
                    fake_warped_images_tex_in = fake_warped_images_tex_in * (1-warped_clothing_items_mask)
                    fake_mask = fake_mask * (1-warped_clothing_items_mask)
                    tex_parsings = util.removeClothingItems(tex_parsings, warped_clothing_items_mask)

                if self.no_neck_tex:
                    fake_warped_images_tex_in = fake_warped_images_tex_in * (1-warped_neck_mask)
                    fake_mask = fake_mask * (1-warped_neck_mask)
                    tex_parsings = util.removeNeck(tex_parsings, warped_neck_mask)

            # downsample texture inputs if necessary
            if self.downsample_tex:
                fake_warped_images_tex_in = self.downsampler(fake_warped_images_tex_in)
                if self.no_background and self.mask_gen_outputs:
                    if self.gan_mode == 'flow_and_texture':
                        fake_mask = self.downsampler(fake_mask)
                    if self.gan_mode == 'texture_only':
                        self.masks = self.downsampler(self.masks)

                    if self.no_clothing_items:
                        warped_clothing_items_mask = self.downsampler(warped_clothing_items_mask)
                    if self.no_neck_tex:
                        warped_neck_mask = self.downsampler(warped_neck_mask)

                if self.use_parsings_tex_in:
                    tex_parsings = self.downsampler(tex_parsings)
                if self.use_encoding_net:
                    warped_expanded_parsings = self.downsampler(warped_expanded_parsings).round()
                self.gen_conditions = self.downsampler(self.gen_conditions)
                self.rec_conditions = self.downsampler(self.rec_conditions)
                self.real_disc_conditions = self.downsampler(self.real_disc_conditions)
                self.fake_disc_conditions = self.downsampler(self.fake_disc_conditions)

            # add parsings to texture input
            if self.use_encoding_net:
                if self.use_avg_features:
                    labels = warped_expanded_parsings
                else:
                    labels = None

                # we need to downsample the conditions once more
                # since the encoder has 4 downsampling layers while the generator has 3
                if self.is_G_local:
                    enc_cond = self.gen_conditions[0]
                else:
                    enc_cond = self.downsampler(self.gen_conditions)

                features = self.netE_tex(fake_warped_images_tex_in, enc_cond, labels)
                if self.use_parsings_tex_in:
                    tex_in = torch.cat((features, warped_expanded_parsings), 1)
                    init_tex_in = torch.cat((fake_warped_images_tex_in, tex_parsings), 1)
                else:
                    tex_in = features
                    init_tex_in = fake_warped_images_tex_in

            elif self.use_parsings_tex_in:
                # if self.no_facial_hair: # only the flow network ignores facial hair
                #     tex_parsings = util.restoreFacialHair(tex_parsings, warped_facial_hair_mask)

                tex_in = torch.cat((fake_warped_images_tex_in, tex_parsings),1)
                # else:
                # tex_in = torch.cat((fake_warped_images_tex_in, fake_parsings),1)
            else:
                tex_in = fake_warped_images_tex_in

            # forward pass of texture net
            if 'ada' in self.netG_tex_arch:
                tex_in_id_features, tex_in_age_features = self.netG_tex.encode(tex_in, fake_mask)
                generated_tex_images = self.netG_tex.decode(tex_in_id_features, tex_in_age_features, self.gen_conditions)
            else:
                generated_tex_images = self.netG_tex(tex_in, self.gen_conditions)

            if self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                fake_flow_images = torch.cat((fake_parsings, fake_landmarks),1)
            elif self.use_parsings or (self.use_landmarks and self.json_landmarks):
                fake_flow_images = fake_parsings
            elif self.use_landmarks and (not self.json_landmarks):
                fake_flow_images = fake_landmarks
            else:
                fake_flow_images = fake_warped_images

            if self.no_background and self.mask_gen_outputs:
                if self.no_clothing_items:
                    fake_mask = fake_mask * (1-warped_clothing_items_mask)
                    # orig_mask = self.masks * (1-self.clothing_items_masks)
                    tex_parsings = util.removeClothingItems(tex_parsings, warped_clothing_items_mask)

                if self.no_neck_tex:
                    fake_mask = fake_mask * (1-warped_neck_mask)
                    # orig_mask = orig_mask * (1-self.neck_masks)
                    tex_parsings = util.removeNeck(tex_parsings, warped_neck_mask)

                generated_tex_images = generated_tex_images * fake_mask

            if self.use_parsings_in_disc and not self.use_parsings_tex_out:
                fake_tex_images = torch.cat((generated_tex_images, tex_parsings), 1)
            else:
                fake_tex_images = generated_tex_images

            fake_images = (fake_flow_images, fake_tex_images)

        # Fake Detection and Loss
        pred_flow_fake_pool_gan, pred_flow_fake_pool_class, _, pred_tex_fake_pool_gan, pred_tex_fake_pool_class, _ = self.discriminate(fake_images, use_pool=True)
        if 'flow' in self.gan_mode and not self.flow_fixed:
            if self.selective_class_loss_flow:
                flow_fake_class_target = torch.cat((self.flow_class_B, self.flow_class_A), 0).long().cuda()
                loss_D_fake_flow = self.criterionGAN_flow(pred_flow_fake_pool_gan, flow_fake_class_target, False)
                loss_D_class_fake_flow = 0
            else:
                loss_D_fake_flow = self.criterionGAN_flow(pred_flow_fake_pool_gan, False)
                if self.use_class_loss_flow and self.opt.classify_fakes:
                    flow_fake_class_target = self.Tensor(2 * self.class_A.shape[0]).fill_(self.numFlowClasses).long()
                    loss_D_class_fake_flow = self.criterionClass_flow(pred_flow_fake_pool_class, flow_fake_class_target) * self.opt.lambda_class_flow
                else:
                    loss_D_class_fake_flow = 0
        else:
            loss_D_fake_flow = 0
            loss_D_class_fake_flow = 0
        if 'texture' in self.gan_mode:
            if self.selective_class_loss_tex:
                tex_fake_class_target = torch.cat((self.class_B, self.class_A), 0).long().cuda()
                loss_D_fake_tex = self.criterionGAN_tex(pred_tex_fake_pool_gan, tex_fake_class_target, False)
                loss_D_class_fake_tex = 0
            else:
                loss_D_fake_tex = self.criterionGAN_tex(pred_tex_fake_pool_gan, False)
                if self.use_class_loss_tex and self.opt.classify_fakes:
                    tex_fake_class_target = self.Tensor(2 * self.class_A.shape[0]).fill_(self.numClasses).long()
                    loss_D_class_fake_tex = self.criterionClass_tex(pred_tex_fake_pool_class, tex_fake_class_target) * self.opt.lambda_class_tex
                else:
                    loss_D_class_fake_tex = 0
        else:
            loss_D_fake_tex = 0
            loss_D_class_fake_tex = 0

        loss_D_fake = loss_D_fake_flow * self.opt.lambda_gan_flow + loss_D_fake_tex * self.opt.lambda_gan_tex
        loss_D_class_fake = loss_D_class_fake_flow + loss_D_class_fake_tex
        if self.opt.classify_fakes:
            loss_D_class_fake = loss_D_class_fake / 2

        # Real Detection and Loss
        if 'flow' in self.gan_mode and not self.flow_fixed:
            if self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                flow_real = torch.cat((self.parsings,self.landmarks),1)
            elif self.use_parsings:
                flow_real = self.parsings
            elif self.use_landmarks and (not self.json_landmarks):
                flow_real = self.landmarks
            else:
                flow_real = rgb_in
        else:
            flow_real = None

        if 'texture' in self.gan_mode:
            if self.no_background:
                tex_reals_in = self.masked_reals.detach()
            else:
                tex_reals_in = self.reals.detach()

            if self.no_clothing_items:
                tex_reals_in = tex_reals_in * (1-self.clothing_items_masks)

            if self.no_neck_tex:
                tex_reals_in = tex_reals_in * (1-self.neck_masks)

            if self.use_parsings_tex_out or self.use_parsings_in_disc:
                tex_parsings_real = self.parsings.detach()
                if self.no_clothing_items:
                    tex_parsings_real = util.removeClothingItems(tex_parsings_real, self.clothing_items_masks)

                if self.no_neck_tex:
                    tex_parsings_real = util.removeNeck(tex_parsings_real, self.neck_masks)
                # if self.no_facial_hair:
                #     tex_parsings_real = util.restoreFacialHair(tex_parsings_real, self.facial_hair_masks)

                tex_real = torch.cat((tex_reals_in, tex_parsings_real),1)

                # else:
                #     tex_real = torch.cat((self.masked_reals, self.parsings),1)
            else:
                tex_real = tex_reals_in
        else:
            tex_real = None

        if self.downsample_tex:
            tex_real = self.downsampler(tex_real)

        real_images = (flow_real, tex_real)
        pred_flow_real_gan, pred_flow_real_class, pred_flow_real_feat, pred_tex_real_gan, pred_tex_real_class, pred_tex_real_feat = self.discriminate(real_images)
        if 'flow' in self.gan_mode and not self.flow_fixed:
            if self.selective_class_loss_flow:
                flow_real_class_target = torch.cat((self.flow_class_A, self.flow_class_B), 0).long().cuda()
                loss_D_real_flow = self.criterionGAN_flow(pred_flow_real_gan, flow_real_class_target, True)
                loss_D_class_real_flow = 0
            else:
                loss_D_real_flow = self.criterionGAN_flow(pred_flow_real_gan, True)
                if self.use_class_loss_flow:
                    flow_real_class_target = torch.cat((self.flow_class_A, self.flow_class_B), 0).long().cuda()
                    loss_D_class_real_flow = self.criterionClass_flow(pred_flow_real_class, flow_real_class_target) * self.opt.lambda_class_flow
                else:
                    loss_D_class_real_flow = 0
        else:
            loss_D_real_flow = 0
            loss_D_class_real_flow = 0
        if 'texture' in self.gan_mode:
            if self.selective_class_loss_tex:
                tex_real_class_target = torch.cat((self.class_A, self.class_B), 0).long().cuda()
                loss_D_real_tex = self.criterionGAN_tex(pred_tex_real_gan, tex_real_class_target, True)
                loss_D_class_real_tex = 0
            else:
                loss_D_real_tex = self.criterionGAN_tex(pred_tex_real_gan, True)
                if self.use_class_loss_tex:
                    tex_real_class_target = torch.cat((self.class_A, self.class_B), 0).long().cuda()
                    loss_D_class_real_tex = self.criterionClass_tex(pred_tex_real_class, tex_real_class_target) * self.opt.lambda_class_tex
                else:
                    loss_D_class_real_tex = 0
        else:
            loss_D_real_tex = 0
            loss_D_class_real_tex = 0

        loss_D_real = loss_D_real_flow * self.opt.lambda_gan_flow + loss_D_real_tex * self.opt.lambda_gan_tex
        loss_D_class_real = loss_D_class_real_flow + loss_D_class_real_tex
        if self.opt.classify_fakes:
            loss_D_class_real = loss_D_class_real / 2

        # Compute loss for gradient penalty.
        if self.netD_tex_arch == 'aux':
            bSize = int(fake_tex_images.size(0))
            tex_disc_gp_real_inputs = tex_real.detach()
            tex_disc_gp_fake_inputs = fake_tex_images.detach()
            alpha = torch.rand(bSize, 1, 1, 1).cuda()
            real_fake_interp = (alpha * tex_disc_gp_real_inputs + (1 - alpha) * tex_disc_gp_fake_inputs).requires_grad_(True)
            interp_out, _ = self.netD_tex(real_fake_interp)
            loss_D_gp = self.gradient_penalty(interp_out, real_fake_interp)
        else:
            loss_D_gp = 0

        # GAN loss (Fake Passability Loss)
        if 'flow' in self.gan_mode and not self.flow_fixed:
            if self.use_class_loss_flow and not self.add_disc_cond_flow_class or self.selective_class_loss_flow:
                flow_disc_inputs = fake_flow_images
            else:
                flow_disc_inputs = torch.cat((fake_flow_images, self.flow_fake_disc_conditions), 1)

            pred_flow_fake_gan, pred_flow_fake_class, pred_flow_fake_feat = self.netD_flow(flow_disc_inputs)
            if self.selective_class_loss_flow:
                flow_gen_fake_class_target = torch.cat((self.flow_class_B, self.flow_class_A), 0).long().cuda()
                loss_G_GAN_flow = self.criterionGAN_flow(pred_flow_fake_gan, flow_gen_fake_class_target, True, is_gen=True)
                loss_G_GAN_class_flow = 0
            else:
                loss_G_GAN_flow = self.criterionGAN_flow(pred_flow_fake_gan, True)
                if self.use_class_loss_flow:
                    flow_gen_fake_class_target = torch.cat((self.flow_class_B, self.flow_class_A), 0).long().cuda()
                    loss_G_GAN_class_flow = self.criterionClass_flow(pred_flow_fake_class, flow_gen_fake_class_target) * self.opt.lambda_class_flow
                else:
                    loss_G_GAN_class_flow = 0
        else:
            loss_G_GAN_flow = 0
            loss_G_GAN_class_flow = 0

        if 'texture' in self.gan_mode and (self.netD_tex_arch != 'aux' or (self.netD_tex_arch == 'aux' and optG_wgan)):
            if self.netD_tex_arch == 'aux' or (self.use_class_loss_tex and not self.add_disc_cond_tex_class) or self.selective_class_loss_tex or self.per_class_netD_tex:
                tex_disc_inputs = fake_tex_images
            else:
                tex_disc_inputs = torch.cat((fake_tex_images, self.fake_disc_conditions), 1)

            if self.per_class_netD_tex:
                disc_classes = torch.cat((self.class_B,self.class_A),0).tolist()
                pred_tex_fake_gan, pred_tex_fake_class, pred_tex_fake_feat = self.netD_tex(tex_disc_inputs, disc_classes)
            else:
                pred_tex_fake_gan, pred_tex_fake_class, pred_tex_fake_feat = self.netD_tex(tex_disc_inputs)

            if self.selective_class_loss_tex:
                tex_gen_fake_class_target = torch.cat((self.class_B, self.class_A), 0).long().cuda()
                loss_G_GAN_tex = self.criterionGAN_tex(pred_tex_fake_gan, tex_gen_fake_class_target, True, is_gen=True)
                loss_G_GAN_class_tex = 0
            else:
                loss_G_GAN_tex = self.criterionGAN_tex(pred_tex_fake_gan, True)
                if self.use_class_loss_tex:
                    tex_gen_fake_class_target = torch.cat((self.class_B, self.class_A), 0).long().cuda()
                    loss_G_GAN_class_tex = self.criterionClass_tex(pred_tex_fake_class, tex_gen_fake_class_target) * self.opt.lambda_class_tex
                else:
                    loss_G_GAN_class_tex = 0
        else:
            loss_G_GAN_tex = 0
            loss_G_GAN_class_tex = 0

        loss_G_GAN = loss_G_GAN_flow * self.opt.lambda_gan_flow + loss_G_GAN_tex * self.opt.lambda_gan_tex
        loss_G_GAN_class = loss_G_GAN_class_flow + loss_G_GAN_class_tex

        # GAN feature matching loss
        # loss_G_GAN_Feat = 0
        # if not self.opt.no_ganFeat_loss:
        #     feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        #     D_weights = 1.0 / self.opt.num_D
        #     for i in range(self.opt.num_D):
        #         for j in range(len(pred_fake[i])-1):
        #             loss_G_GAN_Feat += D_weights * feat_weights * \
        #                 self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        bSize = self.masked_reals.shape[0] // 2
        if 'flow' in self.gan_mode and self.getFinalFeat_flow:
            loss_G_GAN_Feat_flow = 0
            for i in range(self.opt.num_D_flow):
                curr_flow_weight = 2 ** (i+1-self.opt.num_D_flow)
                curr_flow_target = torch.cat((pred_flow_real_feat[i][bSize:,:,:,:].detach(),pred_flow_real_feat[i][:bSize,:,:,:].detach()), 0)
                loss_G_GAN_Feat_flow += curr_flow_weight * self.criterionFeat(pred_flow_fake_feat[i], curr_flow_target)
        else:
            loss_G_GAN_Feat_flow = torch.zeros(1).cuda()

        if 'texture' in self.gan_mode and self.getFinalFeat_tex:
            for i in range(self.opt.num_D_tex):
                curr_tex_weight = 2 ** (i+1-self.opt.num_D_tex)
                curr_tex_target = torch.cat((pred_tex_real_feat[i][bSize:,:,:,:].detach(),pred_tex_real_feat[i][:bSize,:,:,:].detach()), 0)
                loss_G_GAN_Feat_tex += curr_tex_weight * self.criterionFeat(pred_tex_fake_feat[i], curr_tex_target)
        else:
            loss_G_GAN_Feat_tex = torch.zeros(1).cuda()

        loss_G_GAN_Feat = (loss_G_GAN_Feat_flow + loss_G_GAN_Feat_tex) * self.opt.lambda_feat

        # Cycle loss
        # currently only supports batch size of 1
        same_class = (self.class_A == self.class_B).tolist()[0]
        same_flow_class = (self.flow_class_A == self.flow_class_B).tolist()[0]

        if self.gan_mode == 'flow_only':
            if same_flow_class:
                rec_flow_images = fake_flow_images
                fake_rec_parsings = fake_parsings
                fake_rec_landmarks = fake_landmarks
                fake_rec_grid = fake_grid
                fake_rec_mask = fake_mask
                if self.use_xy:
                    # rec_input = warped_xy
                    # rec_reference = self.xy
                    rec_input = warped_xy
                    rec_reference = self.xy
                    loss_G_Cycle_flow = self.criterionCycle(rec_input, rec_reference) * self.opt.lambda_cyc_flow
                elif self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                    loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, torch.cat((self.parsings,self.landmarks),1)) * self.opt.lambda_cyc_flow
                elif self.use_parsings:
                    loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, self.parsings) * self.opt.lambda_cyc_flow
                elif self.use_landmarks and (not self.json_landmarks):
                    loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, self.landmarks) * self.opt.lambda_cyc_flow
                else:
                    loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, rgb_in) * self.opt.lambda_cyc_flow
            else:
                if self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                    rec_in_flow_images = fake_warped_images
                    rec_in_parsings = fake_flow_images[:,:3,:,:]
                    rec_in_landmarks = fake_flow_images[:,3:,:,:]
                elif self.use_parsings:
                    rec_in_flow_images = fake_warped_images
                    rec_in_parsings = fake_flow_images
                    rec_in_landmarks = None
                elif self.use_landmarks and (not self.json_landmarks):
                    rec_in_flow_images = fake_warped_images
                    rec_in_parsings = None
                    rec_in_landmarks = fake_flow_images
                else:
                    rec_in_flow_images = fake_flow_images
                    rec_in_parsings = None
                    rec_in_landmarks = None

                if not self.no_rec_flow:
                    rec_net_flow = getattr(self, 'netR_flow')
                else:
                    rec_net_flow = getattr(self, 'netG_flow')

                fake_rec_images, fake_masked_rec_images, fake_rec_grid, fake_rec_mask, \
                fake_rec_parsings, _, _, _, _, fake_rec_landmarks, rec_xy = \
                rec_net_flow(rec_in_flow_images, self.flow_rec_conditions, parsing=rec_in_parsings, \
                                     mask=fake_mask, landmarks=rec_in_landmarks, xy=warped_xy, \
                                     use_rgbxy_inputs=self.use_rgbxy_flow_inputs, use_xy_inputs=self.use_xy_flow_inputs)

                if self.use_xy:
                    # rec_input = rec_xy
                    # rec_reference = self.xy
                    rec_input = rec_xy
                    rec_reference = self.xy
                    loss_G_Cycle_flow = self.criterionCycle(rec_input, rec_reference) * self.opt.lambda_cyc_flow
                if self.use_parsings and self.use_landmarks and (not self.json_landmarks): #fake_parsings is None:
                    rec_flow_images = torch.cat((fake_rec_parsings,fake_rec_landmarks),1)
                    loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, torch.cat((self.parsings,self.landmarks),1)) * self.opt.lambda_cyc_flow
                elif self.use_parsings:#fake_parsings is None:
                    rec_flow_images = fake_rec_parsings
                    loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, self.parsings) * self.opt.lambda_cyc_flow
                elif self.use_landmarks and (not self.json_landmarks):#fake_parsings is None:
                    rec_flow_images = fake_rec_landmarks
                    loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, self.landmarks) * self.opt.lambda_cyc_flow
                else:
                    rec_flow_images = fake_rec_images
                    loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, rgb_in) * self.opt.lambda_cyc_flow

            loss_G_Cycle_tex = 0

        elif self.gan_mode == 'texture_only' and (self.netD_tex_arch != 'aux' or (self.netD_tex_arch == 'aux' and optG_wgan)):
            if same_class:
                rec_tex_images = generated_tex_images
                if self.use_encoding_net:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_ref = rgb_in
                    else:
                        tex_ref = init_tex_in
                else:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_ref = rgb_in
                    else:
                        tex_ref = tex_in
            else:
                if self.use_encoding_net:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_rec_in = torch.cat((generated_tex_images, tex_parsings), 1)
                        tex_ref = rgb_in
                    else:
                        tex_rec_in = generated_tex_images
                        tex_ref = init_tex_in
                else:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_rec_in = torch.cat((generated_tex_images, tex_parsings), 1)
                        tex_ref = rgb_in
                    else:
                        tex_rec_in = generated_tex_images
                        tex_ref = tex_in

                if not self.no_rec_tex:
                    rec_net_tex = getattr(self, 'netR_tex')
                else:
                    rec_net_tex = getattr(self, 'netG_tex')

                if self.no_background and self.mask_gen_outputs:
                    # need to restore background for second forward pass
                    if not same_class:
                        tex_rec_in = tex_rec_in + (1-self.masks) * self.reals

                    # make sure that the reference image is masked
                    tex_ref = tex_ref * self.masks
                    correction = self.masks.numel() / (self.masks > 0.5).sum().float()
                else:
                    correction = 1

                if 'ada' in self.netG_tex_arch:
                    # self.masks is a very naive thing, there should probably be something smarter here instead
                    fake_tex_id_features, fake_tex_age_features = rec_net_tex.encode(tex_rec_in, self.masks)
                    rec_tex_images = rec_net_tex.decode(fake_tex_id_features, fake_tex_age_features, tex_in_age_features)
                else:
                    rec_tex_images = rec_net_tex(tex_rec_in, self.rec_conditions)

                if self.no_background and self.mask_gen_outputs:
                    # mask the generator output
                    rec_tex_images = rec_tex_images * self.masks

            loss_G_Cycle_flow = 0
            loss_G_Cycle_tex = self.criterionCycle(rec_tex_images, tex_ref) * self.opt.lambda_cyc_tex

        else:  # if self.gan_mode == 'flow_and_texture':
            if same_class:
                rec_tex_images = generated_tex_images
                if self.use_encoding_net:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_ref = fake_warped_images_tex_in
                    else:
                        tex_ref = init_tex_in
                else:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_ref = fake_warped_images_tex_in
                    else:
                        tex_ref = tex_in
            else:
                if self.use_encoding_net:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_rec_in = torch.cat((generated_tex_images, tex_parsings), 1)
                        tex_ref = fake_warped_images_tex_in
                    else:
                        tex_rec_in = generated_tex_images
                        tex_ref = init_tex_in
                else:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_rec_in = torch.cat((generated_tex_images, tex_parsings.detach()), 1)
                        tex_ref = fake_warped_images_tex_in
                    else:
                        tex_rec_in = generated_tex_images
                        tex_ref = tex_in

                if self.no_background and self.mask_gen_outputs:
                    # need to restore background for second forward pass
                    if not same_class:
                        tex_rec_in = tex_rec_in + (1-fake_mask) * self.reals

                    # make sure that the reference image is masked
                    tex_ref = tex_ref * fake_mask
                    correction = fake_mask.numel() / (fake_mask > 0.5).sum().float()
                else:
                    correction = 1

                if not self.no_rec_tex:
                    rec_net_tex = getattr(self, 'netR_tex')
                else:
                    rec_net_tex = getattr(self, 'netG_tex')

                if 'ada' in self.netG_tex_arch:
                    # fake_mask is a very naive thing, there should probably be something smarter here instead
                    fake_tex_id_features, fake_tex_age_features = rec_net_tex.encode(tex_rec_in, fake_mask)
                    rec_tex_images = rec_net_tex.decode(fake_tex_id_features, fake_tex_age_features, tex_in_age_features)
                else:
                    rec_tex_images = rec_net_tex(tex_rec_in, self.rec_conditions)

                if self.no_background and self.mask_gen_outputs:
                    # mask the generator output
                    rec_tex_images = rec_tex_images * fake_mask

            #pytorch 0.4.1
            tex_rec_target = tex_ref.detach()

            # in this case, the target_image should be downsampled too
            # if self.downsample_tex and self.use_parsings_tex_in and not self.use_parsings_tex_out:
            #     tex_rec_target = self.downsampler(tex_rec_target)

            if (self.netD_tex_arch != 'aux' or (self.netD_tex_arch == 'aux' and optG_wgan)):
                loss_G_Cycle_tex = self.criterionCycle(rec_tex_images, tex_rec_target) * self.opt.lambda_cyc_tex * correction
            else:
                loss_G_Cycle_tex = 0

            if not self.flow_fixed:
                if same_flow_class:
                    rec_flow_images = fake_flow_images
                    fake_rec_parsings = fake_parsings
                    fake_rec_landmarks = fake_landmarks
                    fake_rec_grid = fake_grid
                    fake_rec_mask = fake_mask
                    if self.use_xy:
                        rec_input = warped_xy
                        rec_reference = self.xy
                        loss_G_Cycle_flow = self.criterionCycle(rec_input, rec_reference) * self.opt.lambda_cyc_flow
                    elif self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                        loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, torch.cat((self.parsings,self.landmarks),1)) * self.opt.lambda_cyc_flow
                    elif self.use_parsings:
                        loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, self.parsings) * self.opt.lambda_cyc_flow
                    elif self.use_landmarks and (not self.json_landmarks):
                        loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, self.landmarks) * self.opt.lambda_cyc_flow
                    else:
                        loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, rgb_in) * self.opt.lambda_cyc_flow
                else:
                    if self.downsample_tex: # we downsampled texture, hence it need to be upsampled here
                        rec_in_flow_images = self.upsampler(rec_tex_images[:,:3,:,:], scale_factor=2, mode='bilinear') # it's only channels 0:3 because the other channels are parsings
                    else:
                        rec_in_flow_images = rec_tex_images[:,:3,:,:] # it's only channels 0:3 because the other channels are parsings

                    if self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                        rec_in_parsings = fake_flow_images[:,:3,:,:]
                        rec_in_landmarks = fake_flow_images[:,3:,:,:]
                    elif self.use_parsings:
                        rec_in_parsings = fake_flow_images
                        rec_in_landmarks = None
                    elif self.use_landmarks and (not self.json_landmarks):
                        rec_in_parsings = None
                        rec_in_landmarks = fake_flow_images
                    else:
                        rec_in_parsings = None
                        rec_in_landmarks = None

                    if not self.no_rec_flow:
                        rec_net_flow = getattr(self, 'netR_flow')
                    else:
                        rec_net_flow = getattr(self, 'netG_flow')

                    fake_rec_images, fake_masked_rec_images, fake_rec_grid, fake_rec_mask, fake_rec_parsings, _, _, _, fake_rec_landmarks, rec_xy = \
                    rec_net_flow(rec_in_flow_images, self.flow_rec_conditions, parsing=rec_in_parsings, mask=fake_mask,
                                 facial_hair_mask=warped_facial_hair_mask, clothing_items_mask=warped_facial_hair_mask,
                                 neck_mask=warped_neck_mask, landmarks=rec_in_landmark, xy=warped_xy,
                                 use_rgbxy_inputs=self.use_rgbxy_flow_inputs, use_xy_inputs=self.use_xy_flow_inputs)

                    if self.use_xy:
                        rec_input = rec_xy
                        rec_reference = self.xy
                        loss_G_Cycle_flow = self.criterionCycle(rec_input, rec_reference) * self.opt.lambda_cyc_flow
                    elif self.use_parsings and self.use_landmarks and (not self.json_landmarks):
                        rec_flow_images = torch.cat((fake_rec_parsings,fake_rec_landmarks),1)
                        loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, torch.cat((self.parsings,self.landmarks),1)) * self.opt.lambda_cyc_flow
                    elif self.use_parsings:
                        rec_flow_images = fake_rec_parsings
                        loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, self.parsings) * self.opt.lambda_cyc_flow
                    elif self.use_landmarks and (not self.json_landmarks):
                        rec_flow_images = fake_rec_landmarks
                        loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, self.landmarks) * self.opt.lambda_cyc_flow
                    else:
                        rec_flow_images = fake_rec_images
                        loss_G_Cycle_flow = self.criterionCycle(rec_flow_images, rgb_in) * self.opt.lambda_cyc_flow
            else:
                loss_G_Cycle_flow = 0

        loss_G_Cycle = loss_G_Cycle_flow + loss_G_Cycle_tex

        # ID loss
        loss_G_ID_flow = 0
        loss_G_ID_tex = 0
        if 'flow' in self.gan_mode and self.forward_pass_id_loss:
            if not self.flow_fixed:
                fake_warped_id_images, _, _, _, \
                _, _, _, _, _, _, warped_id_xy = \
                self.netG_flow(self.masked_reals, self.flow_rec_conditions, parsing=self.parsings,
                               xy=self.xy, use_rgbxy_inputs=self.use_rgbxy_flow_inputs,
                               use_xy_inputs=self.use_xy_flow_inputs)

                flow_id_input = warped_id_xy
                flow_id_target = self.xy
                loss_G_ID_flow = self.criterionCycle(flow_id_input, flow_id_target) * self.opt.lambda_id_flow

        if 'texture' in self.gan_mode and (not self.opt.no_id_loss) and (self.netD_tex_arch != 'aux' or (self.netD_tex_arch == 'aux' and optG_wgan)):
            if self.forward_pass_id_loss:
                if self.use_parsings_tex_in:
                    orig_tex_in = torch.cat((rgb_in, self.parsings),1)
                else:
                    orig_tex_in = rgb_in

                tex_id_target = orig_tex_in.detach()

                if 'ada' in self.netG_tex_arch:
                    orig_id_features, orig_age_features = self.netG_tex.encode(orig_tex_in, self.masks)
                    fake_tex_id_images = self.netG_tex.decode(orig_id_features, orig_age_features, self.rec_conditions)
                elif self.use_encoding_net:
                    sys.exit("No support for forward pass identity loss with pix2pix-hd encoder net")
                else:
                    fake_tex_id_images = self.netG_tex(orig_tex_in, self.rec_conditions)

                if self.no_background and self.mask_gen_outputs:
                    if self.gan_mode == 'flow_and_texture':
                        fake_tex_id_images = fake_tex_id_images * fake_mask
                        tex_id_target = tex_id_target * fake_mask
                        correction = fake_mask.numel() / (fake_mask > 0.5).sum().float()
                    elif self.gan_mode == 'texture_only':
                        fake_tex_id_images = fake_tex_id_images * self.masks
                        tex_id_target = tex_id_target * self.masks
                        correction = self.masks.numel() / (self.masks > 0.5).sum().float()
                    else:
                        correction = 1
                else:
                    correction = 1

                loss_G_ID_tex = self.criterionCycle(fake_tex_id_images, tex_id_target) * self.opt.lambda_id_tex * correction
            else:
                if self.use_encoding_net:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_id_target = init_tex_in[:,:3,:,:]
                    else:
                        tex_id_target = init_tex_in
                else:
                    if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                        tex_id_target = tex_in[:,:3,:,:]
                    else:
                        tex_id_target = tex_in

                loss_G_ID_tex = self.criterionCycle(fake_tex_images, tex_id_target) * self.opt.lambda_id_tex

        loss_G_ID = loss_G_ID_flow + loss_G_ID_tex

        # feature consistency losses
        if 'texture' in self.gan_mode and 'ada' in self.netG_tex_arch:
            loss_G_content_reconst = self.content_reconst_criterion(fake_tex_id_features, tex_in_id_features) * self.opt.lambda_content
            loss_G_age_reconst = self.age_reconst_criterion(fake_tex_age_features, self.gen_conditions) * self.opt.lambda_age
            loss_G_age_embedding = self.age_embedding_criterion(orig_age_features, self.orig_age_mean, self.orig_age_sigma) * self.opt.lambda_embedding

        # minflow loss
        if 'flow' in self.gan_mode and not self.flow_fixed:
            if same_flow_class:
                if self.opt.lambda_cyc_flow > 0:
                    loss_minflow = self.criterionMinFlow(fake_grid) * self.opt.lambda_minflow * 2
                else:
                    loss_minflow = self.criterionMinFlow(fake_grid) * self.opt.lambda_minflow
            else:
                if self.opt.lambda_cyc_flow > 0:
                    loss_minflow = (self.criterionMinFlow(fake_grid) + self.criterionMinFlow(fake_rec_grid)) * self.opt.lambda_minflow
                else:
                    loss_minflow = self.criterionMinFlow(fake_grid) * self.opt.lambda_minflow
        else:
            loss_minflow = 0

        # flow TV loss
        if 'flow' in self.gan_mode and not self.flow_fixed:
            if same_flow_class:
                if self.opt.lambda_cyc_flow > 0:
                    loss_flowTV = self.criterionFlowTV(fake_grid) * self.opt.lambda_flowTV * 2
                else:
                    loss_flowTV = self.criterionFlowTV(fake_grid) * self.opt.lambda_flowTV
            else:
                if self.opt.lambda_cyc_flow > 0:
                    loss_flowTV = (self.criterionFlowTV(fake_grid) + self.criterionFlowTV(fake_rec_grid)) * self.opt.lambda_flowTV
                else:
                    loss_flowTV = self.criterionFlowTV(fake_grid) * self.opt.lambda_flowTV
        else:
            loss_flowTV = 0

        # landmarks loss
        if self.use_landmarks and self.json_landmarks:
            flow_landmark_targets = torch.cat((self.flow_class_B, self.flow_class_A), 0).long().cuda()
            loss_landmarks = self.criterionLandmarks(fake_landmarks,flow_landmark_targets) * self.opt.lambda_landmarks
        else:
            loss_landmarks = 0

        if self.gan_mode == 'flow_and_texture':
            out_warped_images = fake_warped_images_tex_in
        elif self.gan_mode == 'flow_only':
            out_warped_images = fake_masked_warped_images
        else:
            out_warped_images = None

        if self.gan_mode == 'flow_only':
            out_warped_parsings = fake_parsings
        elif self.gan_mode == 'flow_and_texture':
            out_warped_parsings = tex_parsings
        else:
            out_warped_parsings = None

        # upsample texture outputs for visualization
        if self.downsample_tex and infer:
            fake_tex_images = self.upsampler(fake_tex_images, scale_factor=2, mode='bilinear')
            out_warped_images = self.upsampler(out_warped_images, scale_factor=2, mode='bilinear')
            rec_tex_images = self.upsampler(rec_tex_images, scale_factor=2, mode='bilinear')
            if self.use_parsings and self.use_parsings_tex_in:
                out_warped_parsings = self.upsampler(out_warped_parsings, scale_factor=2, mode='nearest')
            if not self.flow_fixed:
                fake_masked_rec_images = self.upsampler(fake_masked_rec_images, scale_factor=2, mode='bilinear')

        return [ self.loss_filter(loss_G_GAN, loss_G_GAN_class, loss_G_GAN_Feat, loss_G_Cycle, loss_G_ID, loss_D_real, loss_D_fake,
                                  loss_D_gp, loss_D_class_real, loss_D_class_fake, loss_minflow, loss_flowTV, loss_landmarks,
                                  loss_G_content_reconst, loss_G_age_reconst, loss_G_age_embedding),
                 None if not (infer and 'flow' in self.gan_mode) else out_warped_images,
                 None if not (infer and 'texture' in self.gan_mode) else fake_tex_images[:,0:3,:,:], # in case there are texture output parsings
                 None if not (infer and 'texture' in self.gan_mode and self.use_parsings_tex_out) else fake_tex_images[:,3:6,:,:], # in case there are texture output parsings
                 None if not (infer and 'texture' in self.gan_mode) else rec_tex_images[:,0:3,:,:], # in case there are output parsings
                 None if not (infer and 'texture' in self.gan_mode and self.use_parsings_tex_out) else rec_tex_images[:,3:6,:,:], # in case there are texture output parsings
                 None if not (infer and 'flow' in self.gan_mode and not self.flow_fixed) else fake_masked_rec_images[:,0:3,:,:], # flow rec images (they are parsings if self.use_parsings is True). it's channels 0:3 in case there are output parsings
                 None if not (infer and 'flow' in self.gan_mode and self.use_parsings) else out_warped_parsings,
                 None if not (infer and 'flow' in self.gan_mode and self.use_parsings and not self.flow_fixed) else fake_rec_parsings,
                 None if not (infer and 'flow' in self.gan_mode and self.use_landmarks and (not self.json_landmarks)) else fake_landmarks,
                 None if not (infer and 'flow' in self.gan_mode and self.use_landmarks and (not self.json_landmarks) and not self.flow_fixed) else fake_rec_landmarks,
                 None if not (infer and 'flow' in self.gan_mode) else fake_grid,
                 None if not (infer and 'flow' in self.gan_mode and not self.flow_fixed) else fake_rec_grid]

    def inference(self, data):
        # Encode Inputs
        self.encode_input(data, mode='test')
        if self.isEmpty:
            return

        self.numValid = self.valid.sum().item()
        sz = self.reals.size()

        if self.original_munit:
            # remove clothing items from texture input
            if self.no_background and not self.mask_gen_outputs:
                rgb_in = self.masked_reals
            else:
                rgb_in = self.reals

            if self.no_clothing_items:
                rgb_in = rgb_in * (1-self.clothing_items_masks)
                self.masks = self.masks * (1-self.clothing_items_masks)

            if self.no_neck_tex:
                rgb_in = rgb_in * (1-self.neck_masks)
                self.masks = self.masks * (1-self.neck_masks)

            if self.numValid > 1:
                rgb_in_A = rgb_in[:1,:,:,:]
                rgb_in_B = rgb_in[1:,:,:,:]
                mask_A = self.masks[:1,:,:,:]
                mask_B = self.masks[1:,:,:,:]
            elif self.class_A == 0:
                rgb_in_A = rgb_in[:1,:,:,:]
                mask_A = self.masks[:1,:,:,:]
                rgb_in_B = None
                mask_B = None
            else:
                rgb_in_A = None
                mask_A = None
                rgb_in_B = rgb_in[:1,:,:,:]
                mask_B = self.masks[:1,:,:,:]

        # pytorch 0.4.1
        if 'flow' in self.gan_mode:
            if self.use_parsings:
                self.fake_parsings = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
                self.rec_parsings = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
            else:
                self.fake_parsings = None
                self.rec_parsings = None

            if self.use_landmarks and (not self.json_landmarks):
                self.fake_landmarks = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
                self.rec_landmarks = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
            else:
                self.fake_landmarks = None
                self.rec_landmarks = None

            if self.use_xy or self.use_rgbxy_flow_inputs:
                self.warped_xy = self.Tensor(self.numClasses, sz[0], 2, sz[2], sz[3])
                self.rec_xy = self.Tensor(self.numClasses, sz[0], 2, sz[2], sz[3])
            else:
                self.warped_xy = None
                self.rec_xy = None

            self.fake_B_flow = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
            self.rec_A_flow = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
            self.fake_grid = self.Tensor(self.numClasses, sz[0], 2, sz[2], sz[3])
            self.rec_grid = self.Tensor(self.numClasses, sz[0], 2, sz[2], sz[3])
            fake_mask = self.Tensor(sz[0], sz[1], sz[2], sz[3])
            fake_B_flow_rec_input = self.Tensor(sz[0], 3, sz[2], sz[3])

            # create original grid (equivalent to numpy meshgrid)
            x = torch.linspace(-1, 1, steps=sz[3]).type_as(self.Tensor(0))
            y = torch.linspace(-1, 1, steps=sz[2]).type_as(self.Tensor(0))
            xx = x.view(1, -1).repeat(sz[0], 1, sz[2], 1)
            yy = y.view(-1, 1).repeat(sz[0], 1, 1,sz[3])
            orig_grid = torch.cat([xx, yy], 1)

        if self.gan_mode == 'seg_and_texture':
            if self.use_expanded_parsings:
                self.fake_B_seg = self.Tensor(self.numClasses, sz[0], self.parsing_labels_num, sz[2], sz[3])
                self.rec_A_seg = self.Tensor(self.numClasses, sz[0], self.parsing_labels_num, sz[2], sz[3])
            else:
                self.fake_B_seg = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
                self.rec_A_seg = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])

            self.fake_B_tex = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
            self.rec_A_tex = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])

        if self.gan_mode == 'texture_only' or self.gan_mode == 'seg_only' or self.gan_mode == 'flow_and_texture':
            # fake_B_tex_rec_input = self.Tensor(sz[0], 3, sz[2], sz[3])
            if self.downsample_tex:
                h_tex = int(sz[2]/2)
                w_tex = int(sz[3]/2)
            else:
                h_tex = sz[2]
                w_tex = sz[3]

            if self.use_parsings_tex_out:
                self.fake_B_tex = self.Tensor(self.numClasses, sz[0], 2*sz[1], h_tex, w_tex)
                self.rec_A_tex = self.Tensor(self.numClasses, sz[0], 2*sz[1], h_tex, w_tex)
            elif 'seg' in self.gan_mode and self.use_expanded_parsings:
                self.fake_B_tex = self.Tensor(self.numClasses, sz[0], self.parsing_labels_num, h_tex, w_tex)
                self.rec_A_tex = self.Tensor(self.numClasses, sz[0], self.parsing_labels_num, h_tex, w_tex)
            else:
                self.fake_B_tex = self.Tensor(self.numClasses, sz[0], sz[1], h_tex, w_tex)
                self.rec_A_tex = self.Tensor(self.numClasses, sz[0], sz[1], h_tex, w_tex)

        with torch.no_grad(): # pytorch 0.4.1, for pytorch 0.3.1 comment this line and untab the code below
            if self.original_munit:
                #encode images
                orig_id_features_A, orig_age_features_A = self.netG_tex_A.encode(rgb_in_A, mask_A)
                orig_id_features_B, orig_age_features_B = self.netG_tex_B.encode(rgb_in_B, mask_B)
                #within domain decode
                reconst_tex_images_A = self.netG_tex_A.decode(orig_id_features_A, None, orig_age_features_A)
                reconst_tex_images_B = self.netG_tex_B.decode(orig_id_features_B, None, orig_age_features_B)
                # generate random gaussians
                target_A = torch.randn(1, self.cond_global_length).cuda()
                target_B = torch.randn(1, self.cond_global_length).cuda()
                #cross domain decode
                generated_tex_images_BA = self.netG_tex_A.decode(orig_id_features_B, None, target_A)
                generated_tex_images_AB = self.netG_tex_B.decode(orig_id_features_A, None, target_B)
                #encode generated
                recon_id_features_B, recon_age_features_A = self.netG_tex_A.encode(generated_tex_images_BA, mask_B)
                recon_id_features_A, recon_age_features_B = self.netG_tex_B.encode(generated_tex_images_AB, mask_A)
                #decode generated
                cyc_tex_images_A = self.netG_tex_A.decode(recon_id_features_A, None, orig_age_features_A)
                cyc_tex_images_B = self.netG_tex_B.decode(recon_id_features_B, None, orig_age_features_B)

                if self.numValid > 1:
                    self.fake_B_tex[0, :, :, :, :] = torch.cat((reconst_tex_images_A, generated_tex_images_BA),0)
                    self.fake_B_tex[1, :, :, :, :] = torch.cat((generated_tex_images_AB, reconst_tex_images_B),0)
                    self.rec_A_tex[0, :, :, :, :] = torch.cat((reconst_tex_images_A, cyc_tex_images_B),0)
                    self.rec_A_tex[1, :, :, :, :] = torch.cat((cyc_tex_images_A, reconst_tex_images_B),0)
                elif self.class_A == 0:
                    self.fake_B_tex[0, :, :, :, :] = reconst_tex_images_A
                    self.fake_B_tex[1, :, :, :, :] = generated_tex_images_AB
                    self.rec_A_tex[0, :, :, :, :] = reconst_tex_images_A
                    self.rec_A_tex[1, :, :, :, :] = cyc_tex_images_A
                else:
                    self.fake_B_tex[0, :, :, :, :] = generated_tex_images_BA
                    self.fake_B_tex[1, :, :, :, :] = reconst_tex_images_B
                    self.rec_A_tex[0, :, :, :, :] = cyc_tex_images_B
                    self.rec_A_tex[1, :, :, :, :] = reconst_tex_images_B

            else:
                if self.traverse or self.deploy:
                    if self.traverse and self.compare_to_trained_outputs:
                        start = self.compare_to_trained_class - self.trained_class_jump
                        end = start + (self.trained_class_jump * 2) * 2 #arange is between [start, end), end is always omitted
                        self.class_B = torch.arange(start, end, step=self.trained_class_jump*2, dtype=self.class_A.dtype)
                    else:
                        self.class_B = torch.arange(self.numClasses, dtype=self.class_A.dtype)
                    self.get_conditions(mode='test')

                    # remove clothing items from texture input
                    tex_parsings = self.parsings.detach()
                    if self.no_background and not self.mask_gen_outputs:
                        gen_in = self.masked_reals
                    else:
                        gen_in = self.reals

                    if self.no_clothing_items:
                        gen_in = gen_in * (1-self.clothing_items_masks)
                        self.masks = self.masks * (1-self.clothing_items_masks)
                        tex_parsings = util.removeClothingItems(tex_parsings, self.clothing_items_masks)

                    if self.no_neck_tex:
                        gen_in = gen_in * (1-self.neck_masks)
                        self.masks = self.masks * (1-self.neck_masks)
                        tex_parsings = util.removeNeck(tex_parsings, self.neck_masks)

                    generator = getattr(self, 'netG_tex')
                    self.tex_in = gen_in
                    within_domain_idx = -1
                    self.fake_B_tex = generator.infer(self.tex_in, None, self.gen_conditions, within_domain_idx, traverse=self.traverse, deploy=self.deploy, interp_step=self.opt.interp_step)

                else:
                    for i in range(self.numClasses):
                        self.class_B = self.Tensor(self.numValid).long().fill_(i)
                        if 'flow' in self.gan_mode:
                            flow_class_B_idx = self.inv_active_flow_classes_mapping[self.tex2flow_mapping[self.active_classes_mapping[i]]]
                            self.flow_class_B = self.Tensor(self.numValid).long().fill_(flow_class_B_idx)

                        self.get_conditions(mode='test')

                        if self.gan_mode == 'flow_only':
                            if self.use_encoding_net_flow:
                                labels = None

                                flow_enc_input = torch.cat((self.parsings,self.landmarks), 1)
                                flow_enc_cond = self.downsampler(self.flow_gen_conditions)
                                flow_features = self.netE_flow(flow_enc_input, flow_enc_cond, labels)
                            else:
                                flow_features = None

                            _, self.fake_B_flow[i, :, :, :, :], fake_grid, fake_mask, \
                            parsing_out, _, fake_clothing_mask, fake_neck_mask, _, landmarks_out, xy_out = \
                            self.netG_flow(self.masked_reals, self.flow_gen_conditions, parsing=self.parsings,
                                           mask=self.masks, clothing_items_mask=self.clothing_items_masks,
                                           neck_mask=self.neck_masks,landmarks=self.landmarks,
                                           expanded_parsings=self.expanded_parsings, features=flow_features,
                                           xy=self.xy, use_rgbxy_inputs=self.use_rgbxy_flow_inputs,
                                           use_xy_inputs=self.use_xy_flow_inputs)

                            self.fake_grid[i, :, :, :, :] = fake_grid #* fake_mask[:,:2,:,:] + orig_grid * (1-fake_mask[:,:2,:,:])
                            if self.use_parsings:
                                self.fake_parsings.data[i, :, :, :, :].copy_(parsing_out.data)
                            if self.use_landmarks and (not self.json_landmarks):
                                self.fake_landmarks.data[i, :, :, :, :].copy_(landmarks_out.data)
                            if self.use_rgbxy_flow_inputs:
                                self.warped_xy.data[i, :, :, :, :].copy_(xy_out.data)

                            fake_B_flow_rec_input.data.copy_(self.fake_B_flow.data[i, :, :, :, :])

                            if not self.no_rec_flow:
                                rec_net_flow = getattr(self, 'netR_flow')
                            else:
                                rec_net_flow = getattr(self, 'netG_flow')

                            _, self.rec_A_flow[i, :, :, :, :], rec_grid, \
                            fake_rec_mask, rec_parsing_out, _, _, _, _, \
                            rec_landmarks_out, rec_xy_out = rec_net_flow(fake_B_flow_rec_input, self.flow_rec_conditions,
                                                        parsing=parsing_out, mask=fake_mask, landmarks=landmarks_out,
                                                        xy=xy_out, use_rgbxy_inputs=self.use_rgbxy_flow_inputs,
                                                        use_xy_inputs=self.use_xy_flow_inputs)

                            if self.use_parsings:
                                self.rec_parsings.data[i,:,:,:,:].copy_(rec_parsing_out.data)
                            if self.use_landmarks and (not self.json_landmarks):
                                self.rec_landmarks.data[i,:,:,:,:].copy_(rec_landmarks_out.data)
                            if self.use_rgbxy_flow_inputs:
                                self.rec_xy.data[i, :, :, :, :].copy_(rec_xy_out.data)

                            self.rec_grid[i, :, :, :, :] = rec_grid #* fake_rec_mask[:,:2,:,:] + orig_grid * (1-fake_rec_mask[:,:2,:,:])

                        elif self.gan_mode == 'seg_and_texture':
                            if self.use_expanded_parsings:
                                seg_gen_in = self.expanded_parsings
                            else:
                                # seg_gen_in = self.parsings
                                seg_gen_in = util.flip_eye_labels(self.parsings)

                            if self.no_background and not self.mask_gen_outputs:
                                tex_gen_in = self.masked_reals
                            else:
                                tex_gen_in = self.reals

                            if self.no_clothing_items:
                                tex_gen_in = tex_gen_in * (1-self.clothing_items_masks)
                                self.masks = self.masks * (1-self.clothing_items_masks)

                            if self.no_neck_tex:
                                tex_gen_in = tex_gen_in * (1-self.neck_masks)
                                self.masks = self.masks * (1-self.neck_masks)

                            if self.use_orig_age_features_within_domain:
                                within_domain_idx = i
                            else:
                                within_domain_idx = -1

                            self.fake_B_seg[i, :, :, :, :] = self.netG_seg.infer(seg_gen_in, None, self.gen_conditions, within_domain_idx)
                            self.fake_B_tex[i, :, :, :, :] = self.netG_tex.infer(tex_gen_in, self.fake_B_seg[i, :, :, :, :], self.gen_conditions, within_domain_idx)

                            fake_B_seg_rec_input = self.fake_B_seg[i, :, :, :, :]
                            fake_B_tex_rec_input = self.fake_B_tex[i, :, :, :, :]

                            self.rec_A_seg[i, :, :, :, :] = self.netG_seg.infer(fake_B_seg_rec_input, None, self.rec_conditions, -1)
                            self.rec_A_tex[i, :, :, :, :] = self.netG_tex.infer(fake_B_tex_rec_input, self.rec_A_seg[i, :, :, :, :], self.rec_conditions, -1)

                        elif self.gan_mode == 'texture_only' or self.gan_mode == 'seg_only':
                            # remove clothing items from texture input
                            tex_parsings = self.parsings.detach()
                            if self.gan_mode == 'seg_only':
                                if self.use_expanded_parsings:
                                    gen_in = self.expanded_parsings
                                else:
                                    gen_in = self.parsings

                                generator = getattr(self, 'netG_seg')
                                rec_net = getattr(self, 'netG_seg')
                                is_adain_gen = 'ada' in self.netG_tex_arch

                            else:
                                if self.no_background and not self.mask_gen_outputs:
                                    gen_in = self.masked_reals
                                else:
                                    gen_in = self.reals

                                if self.no_clothing_items:
                                    gen_in = gen_in * (1-self.clothing_items_masks)
                                    self.masks = self.masks * (1-self.clothing_items_masks)
                                    tex_parsings = util.removeClothingItems(tex_parsings, self.clothing_items_masks)

                                if self.no_neck_tex:
                                    gen_in = gen_in * (1-self.neck_masks)
                                    self.masks = self.masks * (1-self.neck_masks)
                                    tex_parsings = util.removeNeck(tex_parsings, self.neck_masks)

                                generator = getattr(self, 'netG_tex')
                                if not self.no_rec_tex:
                                    rec_net_tex = getattr(self, 'netR_tex')
                                else:
                                    rec_net = getattr(self, 'netG_tex')

                                is_adain_gen = 'ada' in self.netG_tex_arch

                            if self.downsample_tex:
                                gen_in = self.downsampler(gen_in)
                                if self.use_parsings_tex_in:
                                    tex_parsings = self.downsampler(tex_parsings)
                                if self.use_encoding_net:
                                    self.expanded_parsings = self.downsampler(self.expanded_parsings).round()

                                self.gen_conditions = self.downsampler(self.gen_conditions)
                                self.rec_conditions = self.downsampler(self.rec_conditions)

                            # add parsings to texture input
                            if self.use_encoding_net:
                                if self.use_avg_features:
                                    labels = self.expanded_parsings
                                else:
                                    labels = None

                                # we need to downsample the conditions once more
                                # since the encoder has 4 downsampling layers while the generator has 3
                                if self.downsample_tex:
                                    enc_cond = self.downsampler(self.gen_conditions)
                                else:
                                    enc_cond = self.downsampler(self.gen_conditions[0])

                                features = self.netE_tex(self.masked_real, enc_cond, labels)
                                if self.use_parsings_tex_in:
                                    tex_in = torch.cat((features, self.expanded_parsings), 1)
                                    # init_tex_in = torch.cat((self.masked_reals, tex_parsings), 1)
                                else:
                                    tex_in = features
                                    # init_tex_in = self.masked_reals
                            elif self.use_parsings_tex_in:
                                # if self.no_facial_hair:
                                #     tex_parsings = util.restoreFacialHair(tex_parsings, self.facial_hair_masks)

                                tex_in = torch.cat((gen_in, tex_parsings),1)
                            else:
                                tex_in = gen_in

                            self.tex_in = tex_in
                            if is_adain_gen:
                                if self.use_orig_age_features_within_domain:
                                    within_domain_idx = i
                                else:
                                    within_domain_idx = -1
                                if self.isTrain:
                                    self.fake_B_tex[i, :, :, :, :] = self.g_running.infer(tex_in, None, self.gen_conditions, within_domain_idx)
                                else:
                                    self.fake_B_tex[i, :, :, :, :] = generator.infer(tex_in, None, self.gen_conditions, within_domain_idx)
                            else:
                                self.fake_B_tex[i, :, :, :, :] = generator(tex_in, self.gen_conditions)

                            # fake_B_tex_rec_input.data.copy_(self.fake_B_tex.data[i, :, :, :, :]) # previous version
                            if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                                fake_B_tex_rec_input = torch.cat((self.fake_B_tex[i, :, :, :, :], self.parsings), 1)
                            else:
                                fake_B_tex_rec_input = self.fake_B_tex[i, :, :, :, :]

                            if is_adain_gen:
                                # self.masks is a very naive thing, there should probably be something smarter here instead
                                if self.use_orig_age_features_within_domain:
                                    within_domain_idx = i
                                else:
                                    within_domain_idx = -1
                                if self.isTrain:
                                    self.rec_A_tex[i, :, :, :, :] = self.g_running.infer(fake_B_tex_rec_input, None, self.rec_conditions, within_domain_idx)
                                else:
                                    self.rec_A_tex[i, :, :, :, :] = rec_net.infer(fake_B_tex_rec_input, None, self.rec_conditions, within_domain_idx)
                            else:
                                self.rec_A_tex[i, :, :, :, :] = rec_net(fake_B_tex_rec_input, self.rec_conditions)

                            fake_clothing_mask = self.clothing_items_masks
                            fake_neck_mask = self.neck_masks
                        else: # self.gan_mode == 'flow_and_texture'
                            if self.use_encoding_net_flow:
                                labels = None

                                flow_enc_input = torch.cat((self.parsings,self.landmarks), 1)
                                flow_enc_cond = self.downsampler(self.flow_gen_conditions)
                                flow_features = self.netE_flow(flow_enc_input, flow_enc_cond, labels)
                            else:
                                flow_features = None

                            unmasked_fake_B_flow, self.fake_B_flow[i, :, :, :, :], fake_grid, fake_mask, \
                            parsing_out, fake_facial_hair_mask, fake_clothing_items_mask, fake_neck_mask, \
                            warped_expanded_parsings, landmarks_out, xy_out = \
                            self.netG_flow(self.reals, self.flow_gen_conditions, parsing=self.parsings, mask=self.masks,
                                           facial_hair_mask=self.facial_hair_masks, clothing_items_mask=self.clothing_items_masks,
                                           neck_mask=self.neck_masks, expanded_parsings=self.expanded_parsings, landmarks=self.landmarks,
                                           features=flow_features, xy=self.xy, use_rgbxy_inputs=self.use_rgbxy_flow_inputs,
                                           use_xy_inputs=self.use_xy_flow_inputs)

                            # remove clothing items from texture input
                            tex_parsings = parsing_out
                            self.fake_grid[i, :, :, :, :] = fake_grid #* fake_mask[:,:2,:,:] + orig_grid * (1-fake_mask[:,:2,:,:])

                            if self.use_rgbxy_flow_inputs:
                                self.warped_xy.data[i, :, :, :, :].copy_(xy_out.data)

                            if self.no_background and not self.mask_gen_outputs:
                                if self.no_clothing_items:
                                    self.fake_B_flow[i, :, :, :, :] = self.fake_B_flow[i, :, :, :, :] * (1-fake_clothing_items_mask)
                                    fake_mask = fake_mask * (1-fake_clothing_items_mask)
                                    tex_parsings = util.removeClothingItems(tex_parsings, fake_clothing_items_mask)

                                if self.no_neck_tex:
                                    self.fake_B_flow[i, :, :, :, :] = self.fake_B_flow[i, :, :, :, :] * (1-fake_neck_mask)
                                    fake_mask = fake_mask * (1-fake_neck_mask)
                                    tex_parsings = util.removeNeck(tex_parsings, fake_neck_mask)

                            # downsample texture inputs if necessary
                            if self.downsample_tex:
                                downsampled_fake_flow = self.downsampler(self.fake_B_flow[i, :, :, :, :])
                                if self.use_parsings_tex_in:
                                    tex_parsings = self.downsampler(tex_parsings)
                                if self.use_encoding_net:
                                    warped_expanded_parsings = self.downsampler(warped_expanded_parsings).round()

                                self.gen_conditions = self.downsampler(self.gen_conditions)
                                self.rec_conditions = self.downsampler(self.rec_conditions)

                            # add parsings to texture input
                            if self.use_encoding_net:
                                if self.use_avg_features:
                                    labels = warped_expanded_parsings
                                else:
                                    labels = None

                                # we need to downsample the conditions once more
                                # since the encoder has 4 downsampling layers while the generator has 3
                                if self.downsample_tex:
                                    enc_cond = self.downsampler(self.gen_conditions)
                                else:
                                    enc_cond = self.gen_conditions[0]

                                features = self.netE_tex(self.fake_B_flow[i, :, :, :, :], enc_cond, labels)
                                if self.use_parsings_tex_in:
                                    tex_in = torch.cat((features, warped_expanded_parsings), 1)
                                    # init_tex_in = torch.cat((self.masked_reals, tex_parsings), 1)
                                else:
                                    tex_in = features
                                    # init_tex_in = self.masked_reals
                            elif self.use_parsings_tex_in:
                                # if self.no_facial_hair: # only the flow network ignores facial hair
                                #     tex_parsings = util.restoreFacialHair(tex_parsings, fake_facial_hair_mask)

                                tex_in = torch.cat((self.fake_B_flow[i, :, :, :, :], tex_parsings),1)
                                # else:
                                # tex_in = torch.cat((fake_warped_images_tex_in, fake_parsings),1)
                            else:
                                tex_in = self.fake_B_flow[i, :, :, :, :]

                            if 'ada' in self.netG_tex_arch:
                                self.fake_B_tex[i, :, :, :, :] = self.netG_tex(tex_in, fake_mask, self.gen_conditions)
                            else:
                                self.fake_B_tex[i, :, :, :, :] = self.netG_tex(tex_in, self.gen_conditions)

                            # fake_B_tex_rec_input.data.copy_(self.fake_B_tex.data[i, :, :, :, :]) # previous version
                            if self.use_parsings_tex_in and not self.use_parsings_tex_out:
                                fake_B_tex_rec_input = torch.cat((self.fake_B_tex[i, :, :, :, :], tex_parsings.detach()), 1)
                            else:
                                fake_B_tex_rec_input = self.fake_B_tex.data[i, :, :, :, :]

                            if not self.no_rec_tex:
                                rec_net_tex = getattr(self, 'netR_tex')
                            else:
                                rec_net_tex = getattr(self, 'netG_tex')

                            if 'ada' in self.netG_tex_arch:
                                # fake_mask is a very naive thing, there should probably be something smarter here instead
                                self.rec_A_tex[i, :, :, :, :] = rec_net_tex(fake_B_tex_rec_input, fake_mask, self.rec_conditions)
                            else:
                                self.rec_A_tex[i, :, :, :, :] = rec_net_tex(fake_B_tex_rec_input, self.rec_conditions)

                            if self.downsample_tex: # we downsampled texture, hence it need to be upsampled here
                                fake_B_flow_rec_input = self.upsampler(self.rec_A_tex[i, :, :3, :, :], scale_factor=2, mode='bilinear') # it's only channels 0:3 because the other channels are parsings
                                if self.use_parsings_tex_in:
                                    self.fake_parsings.data[i, :, :, :, :] = self.upsampler(tex_parsings, scale_factor=2, mode='bilinear')
                            else:
                                fake_B_flow_rec_input = self.rec_A_tex[i, :, :3, :, :]
                                if self.use_parsings:
                                    self.fake_parsings.data[i, :, :, :, :].copy_(tex_parsings) # = \

                            if self.use_landmarks and (not self.json_landmarks):
                                self.fake_landmarks.data[i, :, :, :, :].copy_(landmarks_out)

                            if not self.no_rec_flow:
                                rec_net_flow = getattr(self, 'netR_flow')
                            else:
                                rec_net_flow = getattr(self, 'netG_flow')

                            _, self.rec_A_flow[i, :, :, :, :], rec_grid, \
                            fake_rec_mask, rec_parsing_out, _, _, _, _, \
                            rec_landmarks_out, rec_xy_out = rec_net_flow(fake_B_flow_rec_input, self.flow_rec_conditions,
                                                                         parsing=parsing_out, mask=fake_mask, landmarks=landmarks_out,
                                                                         xy=xy_out, use_rgbxy_inputs=self.use_rgbxy_flow_inputs,
                                                                         use_xy_inputs=self.use_xy_flow_inputs)

                            self.rec_grid[i, :, :, :, :] = rec_grid * fake_rec_mask[:,:2,:,:] + orig_grid * (1-fake_rec_mask[:,:2,:,:])
                            if self.use_parsings:
                                self.rec_parsings.data[i,:,:,:,:].copy_(rec_parsing_out)
                            if self.use_landmarks and (not self.json_landmarks):
                                self.rec_landmarks.data[i, :, :, :, :].copy_(rec_landmarks_out)
                            if self.use_rgbxy_flow_inputs:
                                self.rec_xy.data[i, :, :, :, :].copy_(rec_xy_out)

                            if self.use_parsings_tex_out:
                                final_mask = ((self.fake_B_tex[i, :, 3:, :, :] > 0).sum(dim=1, keepdim=True) > 0).float().expand(-1,3,-1,-1).contiguous()

                            elif not self.mask_gen_outputs:
                                if self.no_clothing_items and self.no_neck_tex:
                                    background = torch.zeros_like(fake_mask)
                                    valid_output = ((self.fake_B_tex[i, :, :, :, :].abs() > 0.01).sum(dim=1, keepdim=True) > 0).float().expand(-1,3,-1,-1).contiguous()
                                    if self.downsample_tex:
                                        fake_mask = self.downsampler(fake_mask)
                                        fake_clothing_items_mask = self.downsampler(fake_clothing_items_mask)
                                        fake_neck_mask = self.downsampler(fake_neck_mask)
                                        background = self.downsampler(background)

                                    final_mask = fake_mask * (1 - fake_clothing_items_mask) * (1 - fake_neck_mask) * valid_output + background

                                elif self.no_clothing_items:
                                    background = torch.zeros_like(fake_mask)
                                    valid_output = ((self.fake_B_tex[i, :, :, :, :].abs() > 0.01).sum(dim=1, keepdim=True) > 0).float().expand(-1,3,-1,-1).contiguous()
                                    if self.downsample_tex:
                                        fake_mask = self.downsampler(fake_mask)
                                        fake_clothing_items_mask = self.downsampler(fake_clothing_items_mask)
                                        background = self.downsampler(background)

                                    final_mask = fake_mask * (1 - fake_clothing_items_mask) * valid_output + background

                                elif self.no_neck_tex:
                                    background = torch.zeros_like(fake_mask)
                                    valid_output = ((self.fake_B_tex[i, :, :, :, :].abs() > 0.01).sum(dim=1, keepdim=True) > 0).float().expand(-1,3,-1,-1).contiguous()
                                    if self.downsample_tex:
                                        fake_mask = self.downsampler(fake_mask)
                                        fake_neck_mask = self.downsampler(fake_neck_mask)
                                        background = self.downsampler(background)

                                    final_mask = fake_mask * (1 - fake_neck_mask) * valid_output + background

                                else:
                                    valid_output = ((self.fake_B_tex[i, :, :, :, :].abs() > 0.01).sum(dim=1, keepdim=True) > 0).float().expand(-1,3,-1,-1).contiguous()
                                    if self.downsample_tex:
                                        final_mask = self.downsampler(fake_mask) * valid_output
                                    else:
                                        final_mask = fake_mask * valid_output
                            else:
                                final_mask = fake_mask
                                if self.no_clothing_items:
                                    final_mask = final_mask * (1 - fake_clothing_items_mask)

                                if self.no_neck_tex:
                                    final_mask = final_mask * (1 - fake_neck_mask)

                                if self.downsample_tex:
                                    final_mask = self.downsampler(final_mask)

                            if self.downsample_tex:
                                output_fake_B_flow = self.downsampler(unmasked_fake_B_flow)
                            else:
                                output_fake_B_flow = unmasked_fake_B_flow

                            self.fake_B_tex[i, :, :3, :, :] = output_fake_B_flow * (1 - final_mask) + self.fake_B_tex[i, :, :3, :, :] * final_mask

            visuals = self.get_visuals()

        return visuals

    def save(self, which_epoch):
        if self.original_munit:
            self.save_network(self.netG_tex_A, 'G_tex_A', which_epoch, self.gpu_ids)
            self.save_network(self.netG_tex_B, 'G_tex_B', which_epoch, self.gpu_ids)
            self.save_network(self.netD_tex_A, 'D_tex_A', which_epoch, self.gpu_ids)
            self.save_network(self.netD_tex_B, 'D_tex_B', which_epoch, self.gpu_ids)
        else:
            if 'flow' in self.gan_mode:
                self.save_network(self.netG_flow, 'G_flow', which_epoch, self.gpu_ids)
                self.save_network(self.netD_flow, 'D_flow', which_epoch, self.gpu_ids)
                if not self.no_rec_flow:
                    self.save_network(self.netR_flow, 'R_flow', which_epoch, self.gpu_ids)
                if self.use_encoding_net_flow:
                    self.save_network(self.netE_flow, 'E_flow', which_epoch, self.gpu_ids)
            if 'texture' in self.gan_mode:
                self.save_network(self.netG_tex, 'G_tex', which_epoch, self.gpu_ids)
                self.save_network(self.netD_tex, 'D_tex', which_epoch, self.gpu_ids)
                if not self.no_rec_tex:
                    self.save_network(self.netR_tex, 'R_tex', which_epoch, self.gpu_ids)
                if self.use_encoding_net:
                    self.save_network(self.netE_tex, 'E_tex', which_epoch, self.gpu_ids)
                if self.use_moving_avg:
                    self.save_network(self.g_running, 'g_running', which_epoch, self.gpu_ids)
            if 'seg' in self.gan_mode:
                self.save_network(self.netG_seg, 'G_seg', which_epoch, self.gpu_ids)
                if self.use_moving_avg:
                    self.save_network(self.g_running_seg, 'g_running_seg', which_epoch, self.gpu_ids)
                if self.gan_mode == 'seg_only':
                    self.save_network(self.netD_seg, 'D_seg', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = []
        if 'flow' in self.gan_mode and not self.use_pretrained_flow:
            params += list(self.netG_flow.parameters())
            if not self.no_rec_flow:
                params += list(self.netR_flow.parameters())
            if self.use_encoding_net_flow:
                params += list(self.netE_flow.parameters())
        if 'seg' in self.gan_mode and not self.use_pretrained_seg:
            params += list(self.netG_seg.parameters())
        if 'texture' in self.gan_mode:
            params += list(self.netG_tex.parameters())
            if not self.no_rec_tex:
                params += list(self.netR_tex.parameters())
            if self.use_encoding_net:
                params += list(self.netE_tex.parameters())

        self.optimizer_G = self.optimizer(params, lr=self.opt.lr)
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_flow_params(self):
        # after fixing the flow model for a number of iterations, also start finetuning it
        if self.use_pretrained_flow:
            paramsG = []
            paramsG += list(self.netG_flow.parameters())
            if not self.no_rec_flow:
                paramsG += list(self.netR_flow.parameters())

            paramsG += list(self.netG_tex.parameters())
            if not self.no_rec_tex:
                paramsG += list(self.netR_tex.parameters())
            if self.use_encoding_net:
                paramsG += list(self.netE_tex.parameters())

            self.optimizer_G = self.optimizer(paramsG, lr=self.opt.lr)

            paramsD = []
            paramsD += list(self.netD_flow.parameters())
            paramsD += list(self.netD_tex.parameters())
            self.optimizer_D = self.optimizer(paramsD, lr=opt.lr)
            if self.opt.verbose:
                print('------------ Now also finetuning flow model -----------')

            self.use_pretrained_flow = False

    def update_seg_params(self):
        # after fixing the seg model for a number of iterations, also start finetuning it
        if self.use_pretrained_seg:
            paramsG = []
            paramsG += list(self.netG_seg.parameters())
            paramsG += list(self.netG_tex.parameters())
            if not self.no_rec_tex:
                paramsG += list(self.netR_tex.parameters())
            if self.use_encoding_net:
                paramsG += list(self.netE_tex.parameters())

            self.optimizer_G = self.optimizer(paramsG, lr=self.opt.lr)
            #
            # paramsD = []
            # paramsD += list(self.netD_tex.parameters())
            # self.optimizer_D = self.optimizer(paramsD, lr=self.opt.lr)
            if self.opt.verbose:
                print('------------ Now also finetuning seg model -----------')

            self.use_pretrained_seg = False
            self.seg_fixed = False

    def update_learning_rate(self):
        if self.decay_method == 'linear':
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_G.param_groups:
                mult = param_group.get('mult', 1.0)
                param_group['lr'] = lr * mult
            if self.opt.verbose:
                print('update learning rate: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr
        elif self.decay_method == 'step':
            lr = self.old_lr * self.opt.decay_gamma
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_G.param_groups:
                mult = param_group.get('mult', 1.0)
                param_group['lr'] = lr * mult
            if self.opt.verbose:
                print('update learning rate: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr

    def get_visuals(self):
        return_dicts = [OrderedDict() for i in range(self.numValid)]
        if self.gan_mode == 'seg_only' and self.use_expanded_parsings:
            conversion_func = util.parsingLabels2image
        else:
            conversion_func = util.tensor2im

        real_A = util.tensor2im(self.tex_in.data)
        if self.use_parsings:
            parsings_A = util.tensor2im(self.parsings.data)
        if 'flow' in self.gan_mode:
            fake_B_flow = util.tensor2im(self.fake_B_flow.data)

        if ('texture' in self.gan_mode and not self.use_parsings_tex_out) or self.gan_mode == 'seg_only':
            fake_B_tex = conversion_func(self.fake_B_tex.data)
        else:
            fake_B_tex = conversion_func(self.fake_B_tex.data[:,:,:3,:,:])
        if self.gan_mode == 'seg_and_texture':
            if self.use_expanded_parsings:
                fake_B_seg = util.parsingLabels2image(self.fake_B_seg.data[:,:,:3,:,:])
            else:
                fake_B_seg = util.tensor2im(self.fake_B_seg.data[:,:,:3,:,:])
        if self.use_parsings and 'flow' in self.gan_mode:
            fake_parsings = util.tensor2im(self.fake_parsings.data)
        if self.use_landmarks and (not self.json_landmarks) and 'flow' in self.gan_mode:
            landmarks_A = util.tensor2im(self.landmarks.data)
            fake_landmarks = util.tensor2im(self.fake_landmarks.data)
        if self.use_parsings_tex_out:
            fake_B_tex_parsings = util.tensor2im(self.fake_B_tex.data[:,:,3:,:,:])

        if self.debug_mode:
            if 'flow' in self.gan_mode:
                rec_A_flow = util.tensor2im(self.rec_A_flow.data)
                opt_flow = util.flow2im(self.fake_grid.data)
                inv_opt_flow = util.flow2im(self.rec_grid.data)
                if self.use_parsings:
                    # parsings_A = util.tensor2im(self.parsings.data)
                    rec_parsings = util.tensor2im(self.rec_parsings.data)

            if 'texture' in self.gan_mode or self.gan_mode == 'seg_only':
                if self.use_parsings_tex_out:
                    rec_A_tex = conversion_func(self.rec_A_tex.data[:,:,:3,:,:])
                    rec_A_tex_parsings = util.tensor2im(self.rec_A_tex.data[:,:,3:,:,:])
                else:
                    rec_A_tex = conversion_func(self.rec_A_tex.data[:,:,:,:,:])

            if self.gan_mode == 'seg_and_texture':
                if self.use_expanded_parsings:
                    rec_A_seg = util.parsingLabels2image(self.rec_A_seg.data[:,:,:3,:,:])
                else:
                    rec_A_seg = util.tensor2im(self.rec_A_seg.data[:,:,:3,:,:])

            if self.use_landmarks and (not self.json_landmarks) and 'flow' in self.gan_mode:
                rec_landmarks = util.tensor2im(self.rec_landmarks.data)

        if self.numValid == 1:
            real_A = np.expand_dims(real_A, axis=0)
            if self.use_parsings: #and 'flow' in self.gan_mode:
                parsings_A = np.expand_dims(parsings_A, axis=0)
            if self.use_landmarks and (not self.json_landmarks) and 'flow' in self.gan_mode:
                landmarks_A = np.expand_dims(landmarks_A, axis=0)

        for i in range(self.numValid):
            # get the original image and the results for the current samples
            curr_real_A = real_A[i, :, :, :]
            real_A_img = curr_real_A[:, :, :3]
            if 'seg' in self.gan_mode:
                curr_parsing_A = parsings_A[i, :, :, :]
                real_A_parsing = curr_parsing_A[:, :, :3]

            # start with age progression/regression images
            if 'texture' in self.gan_mode or 'seg' in self.gan_mode:
                if self.traverse or self.deploy:
                    curr_fake_B_tex = fake_B_tex
                else:
                    curr_fake_B_tex = fake_B_tex[:, i, :, :, :]

                if self.gan_mode == 'seg_only':
                    orig_dict = OrderedDict([('orig_img_cls_' + str(i), real_A_img),
                                             ('orig_parsing_cls_' + str(i), real_A_parsing)])
                elif self.traverse or self.deploy:
                    orig_dict = OrderedDict([('orig_img', real_A_img)])
                else:
                    # orig_dict = OrderedDict([('orig_img_cls_' + str(i), real_A_img)])
                    orig_dict = OrderedDict([('orig_img_cls_' + str(self.class_A[i].item()), real_A_img)])

                return_dicts[i].update(orig_dict)
                if self.traverse:
                    out_classes = curr_fake_B_tex.shape[0]
                else:
                    out_classes = self.numClasses

                for j in range(out_classes):
                    fake_res_tex = curr_fake_B_tex[j, :, :, :3]
                    fake_dict_tex = OrderedDict([('tex_trans_to_class_' + str(j), fake_res_tex)])
                    return_dicts[i].update(fake_dict_tex)

            if not (self.traverse or self.deploy):
                if self.gan_mode == 'seg_and_texture':
                    curr_fake_B_seg = fake_B_seg[:, i, :, :, :]
                    orig_dict = OrderedDict([('orig_parsing_cls_' + str(i), real_A_parsing)])
                    return_dicts[i].update(orig_dict)
                    for j in range(self.numClasses):
                        fake_res_seg = curr_fake_B_seg[j, :, :, :3]
                        fake_dict_seg = OrderedDict([('seg_trans_to_class_' + str(j), fake_res_seg)])
                        return_dicts[i].update(fake_dict_seg)

                # show flow network warped outputs
                if 'flow' in self.gan_mode:
                    curr_fake_B_flow = fake_B_flow[:, i, :, :, :]
                    orig_dict = OrderedDict([('orig_img1', real_A_img)])
                    return_dicts[i].update(orig_dict)
                    for j in range(self.numClasses):
                        fake_res_flow = curr_fake_B_flow[j, :, :, :3]
                        fake_dict_flow = OrderedDict([('shape_trans_to_class_' + str(j), fake_res_flow)])
                        return_dicts[i].update(fake_dict_flow)

                if not self.fgnet:
                    # show the generated texture parsings if they are present
                    if self.use_parsings and 'flow' in self.gan_mode:
                        curr_parsing_A = parsings_A[i, :, :, :]
                        real_A_parsing = curr_parsing_A[:, :, :3]

                    if 'texture' in self.gan_mode and self.use_parsings_tex_out:
                        curr_fake_B_tex_parsings = fake_B_tex_parsings[:, i, :, :, :]
                        orig_dict = OrderedDict([('orig_parsings1', real_A_parsing)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            fake_res_tex_parsings = curr_fake_B_tex_parsings[j, :, :, :3]
                            fake_dict_tex_parsings = OrderedDict([('tex_parsings_trans_to_class_' + str(j), fake_res_tex_parsings)])
                            return_dicts[i].update(fake_dict_tex_parsings)

                    # show fake parsings
                    if self.use_parsings and 'flow' in self.gan_mode:
                        curr_fake_parsing = fake_parsings[:, i, :, :, :]
                        orig_dict = OrderedDict([('orig_parsings', real_A_parsing)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            fake_parsing_res = curr_fake_parsing[j, :, :, :3]
                            fake_parsing_dict = OrderedDict([('parsing_trans_to_class_' + str(j), fake_parsing_res)])
                            return_dicts[i].update(fake_parsing_dict)

                    if self.use_landmarks and (not self.json_landmarks) and 'flow' in self.gan_mode:
                        curr_landmarks_A = landmarks_A[i, :, :, :]
                        real_A_landmarks = curr_landmarks_A[:, :, :3]

                        # show fake landmarks
                        curr_fake_landmarks = fake_landmarks[:, i, :, :, :]
                        orig_dict = OrderedDict([('orig_landmarks', real_A_landmarks)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            fake_landmarks_res = curr_fake_landmarks[j, :, :, :3]
                            fake_landmarks_dict = OrderedDict([('landmarks_trans_to_class_' + str(j), fake_landmarks_res)])
                            return_dicts[i].update(fake_landmarks_dict)

                if self.debug_mode and (not self.fgnet):
                    if 'flow' in self.gan_mode:
                        # show forward optical flow
                        curr_opt_flow = opt_flow[:, i, :, :, :]
                        orig_dict = OrderedDict([('orig_img4', real_A_img)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            opt_flow_res = curr_opt_flow[j, :, :, :3]
                            fake_opt_flow_dict = OrderedDict([('optical_flow_to_class_' + str(j), opt_flow_res)])
                            return_dicts[i].update(fake_opt_flow_dict)

                    # continue with tex reconstructions
                    if 'texture' in self.gan_mode or self.gan_mode == 'seg_only':
                        curr_rec_A_tex = rec_A_tex[:, i, :, :, :]
                        if self.gan_mode == 'seg_only':
                            orig_dict = OrderedDict([('orig_img2', real_A_img),
                                                     ('orig_parsing2', real_A_parsing)])
                        else:
                            orig_dict = OrderedDict([('orig_img2', real_A_img)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            rec_res_tex = curr_rec_A_tex[j, :, :, :3]
                            rec_dict_tex = OrderedDict([('tex_rec_from_class_' + str(j), rec_res_tex)])
                            return_dicts[i].update(rec_dict_tex)

                    if self.gan_mode == 'seg_and_texture':
                        curr_rec_A_seg = rec_A_seg[:, i, :, :, :]
                        orig_dict = OrderedDict([('orig_parsing2', real_A_parsing)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            rec_res_seg = curr_rec_A_seg[j, :, :, :3]
                            rec_dict_seg = OrderedDict([('seg_rec_from_class_' + str(j), rec_res_seg)])
                            return_dicts[i].update(rec_dict_seg)

                    # continue with flow reconstructions
                    if 'flow' in self.gan_mode:
                        curr_rec_A_flow = rec_A_flow[:, i, :, :, :]
                        orig_dict = OrderedDict([('orig_img3', real_A_img)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            rec_res_flow = curr_rec_A_flow[j, :, :, :3]
                            rec_dict_flow = OrderedDict([('shape_rec_from_class_' + str(j), rec_res_flow)])
                            return_dicts[i].update(rec_dict_flow)

                        # show reconstructed parsings
                        curr_rec_parsing = rec_parsings[:, i, :, :, :]
                        orig_dict = OrderedDict([('orig_parsings2', real_A_parsing)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            rec_parsing_res = curr_rec_parsing[j, :, :, :3]
                            rec_parsing_dict = OrderedDict([('parsing_rec_from_class_' + str(j), rec_parsing_res)])
                            return_dicts[i].update(rec_parsing_dict)

                        # show reconstructed landmarks
                        if self.use_landmarks and (not self.json_landmarks) and 'flow' in self.gan_mode:
                            curr_rec_landmarks = rec_landmarks[:, i, :, :, :]
                            orig_dict = OrderedDict([('orig_parsings2', real_A_landmarks)])
                            return_dicts[i].update(orig_dict)
                            for j in range(self.numClasses):
                                rec_landmarks_res = curr_rec_landmarks[j, :, :, :3]
                                rec_landmarks_dict = OrderedDict([('parsing_rec_from_class_' + str(j), rec_landmarks_res)])
                                return_dicts[i].update(rec_landmarks_dict)

                        # show inverse optical flow
                        curr_inv_opt_flow = inv_opt_flow[:, i, :, :, :]
                        orig_dict = OrderedDict([('orig_img5', real_A_img)])
                        return_dicts[i].update(orig_dict)
                        for j in range(self.numClasses):
                            inv_opt_flow_res = curr_inv_opt_flow[j, :, :, :3]
                            fake_inv_opt_flow_dict = OrderedDict([('optical_flow_from_class_' + str(j), inv_opt_flow_res)])
                            return_dicts[i].update(fake_inv_opt_flow_dict)

        return return_dicts

class InferenceModel(FlowGANHDModel):
    def forward(self, data):
        return self.inference(data)
