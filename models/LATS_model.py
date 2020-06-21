### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import torch.nn as nn
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

class LATS(BaseModel): #Lifetime Age Transformation Synthesis
    def name(self):
        return 'LATS'

    def init_loss_filter(self):
        def loss_filter(g_gan, g_cycle, g_id, d_real, d_fake, grad_penalty, content_reconst, age_reconst):
            return [l for l in (g_gan, g_cycle, g_id, d_real, d_fake, grad_penalty, content_reconst, age_reconst)]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        st()
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
        self.ngf_global = self.ngf

        self.numClasses = opt.numClasses
        self.use_orig_age_features_within_domain = opt.use_orig_age_features_within_domain
        self.use_moving_avg = 'style' in opt.netG and not opt.no_moving_avg

        self.no_cond_noise = opt.no_cond_noise
        style_dim = opt.adain_gen_style_dim * self.numClasses
        self.duplicate = opt.adain_gen_style_dim

        self.cond_length = style_dim

        self.active_classes_mapping = opt.active_classes_mapping
        self.inv_active_flow_classes_mapping = opt.inv_active_flow_classes_mapping

        if not self.isTrain:
            self.fgnet = opt.fgnet
            self.debug_mode = opt.debug_mode
        else:
            self.fgnet = False
            self.debug_mode = False

        ##### define networks
        # Generators
        self.netG_tex_arch = opt.netG
        tex_norm = opt.norm
        self.netG_tex = self.parallelize(networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, self.numClasses, False,
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
            self.g_running = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, self.numClasses, False,
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

        # Discriminator network
        if self.isTrain:
            self.selective_class_loss_tex = opt.selective_class_loss_tex
            self.selective_class_type_tex = opt.selective_class_type_tex

            self.use_orig_within_domain = opt.use_orig_within_domain
            self.orig_age_features_rec_penalty = opt.orig_age_features_rec_penalty

            self.netD_tex_arch = opt.netD
            n_layers_D_tex = opt.n_layers_D

            if 'ada' in opt.netG:
                use_norm_D_tex = opt.use_norm_D_tex
            else:
                use_norm_D_tex = True

            self.netD_tex = self.parallelize(networks.define_D(opt.output_nc, opt.ndf, opt.netD, numClasses=self.numClasses,
                                             n_layers_D=n_layers_D_tex, norm=tex_norm, use_sigmoid=use_sigmoid,
                                             num_D=opt.num_D_tex, num_init_downsample=self.num_init_downsample_tex,
                                             getIntermFeat=not opt.no_ganIntermFeat, getFinalFeat=self.getFinalFeat_tex,
                                             use_class_head=self.use_class_loss_tex, selective_class_loss=self.selective_class_loss_tex,
                                             classify_fakes=self.opt.classify_fakes, use_disc_cond_with_class=self.add_disc_cond_tex_class,
                                             use_norm=use_norm_D_tex, gpu_ids=self.gpu_ids, init_type='kaiming'))#, activation=opt.activation)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if (not self.isTrain) or (self.isTrain and opt.continue_train) else opt.load_pretrain
            if self.isTrain:
                self.load_network(self.netG_tex, 'G_tex', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_tex, 'D_tex', opt.which_epoch, pretrained_path)
                if self.use_moving_avg:
                    self.load_network(self.g_running, 'g_running', opt.which_epoch, pretrained_path)
            elif self.use_moving_avg:
                self.load_network(self.netG_tex, 'g_running', opt.which_epoch, pretrained_path)
            else:
                self.load_network(self.netG_tex, 'G_tex', opt.which_epoch, pretrained_path)


        # set loss functions and optimizers
        if self.isTrain:
            # define loss functions
            self.loss_filter = self.init_loss_filter('ada' in opt.netG, 'ada' in opt.netG)


            self.criterionGAN = self.parallelize(networks.SelectiveClassesNonSatGANLoss())
            self.R1_reg = networks.R1_reg()

            self.age_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.content_reconst_criterion = self.parallelize(networks.FeatureConsistency())

            self.criterionCycle = self.parallelize(networks.FeatureConsistency()) #torch.nn.L1Loss()
            self.criterionID = self.parallelize(networks.FeatureConsistency()) #torch.nn.L1Loss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_Cycle','G_ID','D_real', 'D_fake', 'Grad_penalty',
                                               'Content_reconst', 'Age_reconst')

            # initialize optimizers
            self.old_lr = opt.lr
            self.decay_method = opt.decay_method

            # set optimization algorithm
            self.optimizer = self.get_optim_alg(opt)

            # optimizer G
            paramsG = []
            params_dict_G_tex = dict(self.netG_tex.named_parameters())
            for key, value in params_dict_G_tex.items():
                decay_cond = ('decoder.mlp' in key)
                if opt.decay_adain_affine_layers:
                    decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                if decay_cond:
                    paramsG += [{'params':[value],'lr':opt.lr * 0.01,'mult':0.01}]
                else:
                    paramsG += [{'params':[value],'lr':opt.lr}]

            self.optimizer_G = self.optimizer(paramsG, lr=opt.lr)

            # optimizer D
            paramsD = list(self.netD_tex.parameters())
            self.optimizer_D = self.optimizer(paramsD, lr=opt.lr)

    def parallelize(self, model):
        if self.isTrain and len(self.gpu_ids) > 0:
            return networks._CustomDataParallel(model)
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
            real_A = data['A']
            real_B = data['B']

            self.class_A = data['A_class']
            self.class_B = data['B_class']

            self.reals = torch.cat((real_A, real_B), 0)

            if len(self.gpu_ids) > 0:
                self.reals = self.reals.cuda()

        else:
            inputs = data['Imgs']
            if inputs.dim() > 4:
                inputs = inputs.squeeze(0)

            self.class_A = data['Classes']
            if self.class_A.dim() > 1:
                self.class_A = self.class_A.squeeze(0)

            if torch.is_tensor(data['Valid']):
                self.valid = data['Valid'].bool()
            else:
                self.valid = torch.ones(1, dtype=torch.bool)

            if self.valid.dim() > 1:
                self.valid = self.valid.squeeze(0)

            if isinstance(data['Paths'][0], tuple):
                self.image_paths = [path[0] for path in data['Paths']]
            else:
                self.image_paths = data['Paths']

            self.isEmpty = False if any(self.valid) else True
            if not self.isEmpty:
                available_idx = torch.arange(len(self.class_A))
                select_idx = torch.masked_select(available_idx, self.valid).long()
                inputs = torch.index_select(inputs, 0, select_idx)

                self.class_A = torch.index_select(self.class_A, 0, select_idx)
                self.image_paths = [val for i, val in enumerate(self.image_paths) if self.valid[i] == 1]

            self.reals = inputs

            if len(self.gpu_ids) > 0:
                self.reals = real_A.cuda()


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

        is_adain_gen = True

        #tex condition mapping
        condG_A_global = self.Tensor(nb, self.cond_length)
        condG_B_global = self.Tensor(nb, self.cond_length)
        condG_A_orig = self.Tensor(nb, self.cond_length)
        condG_B_orig = self.Tensor(nb, self.cond_length)

        if self.no_cond_noise:
            noise_sigma = 0
        else:
            noise_sigma = 0.2

        for i in range(nb):
            condG_A_global[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
            condG_A_global[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1
            if not (self.traverse or self.deploy):
                condG_B_global[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_global[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                condG_A_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_A_orig[i, self.class_A[i]*self.duplicate:(self.class_A[i] + 1)*self.duplicate] += 1

                condG_B_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_orig[i, self.class_B[i]*self.duplicate:(self.class_B[i] + 1)*self.duplicate] += 1

        if mode == 'train':
            if is_adain_gen:
                self.gen_conditions =  torch.cat((condG_A_global, condG_B_global), 0) #torch.cat((self.class_B, self.class_A), 0)
                self.rec_conditions = torch.cat((condG_B_global, condG_A_global), 0)
                self.orig_conditions = torch.cat((condG_A_orig, condG_B_orig),0)
        else:
            self.gen_conditions = condG_A_global #self.class_B
            if not (self.traverse or self.deploy):
                self.rec_conditions = condG_B_global #self.class_A
                self.orig_conditions = condG_A_orig

    def update_G(self, infer=False):
        self.optimizer_G.zero_grad()

        self.get_conditions()
        gen_in = self.reals
        generator = getattr(self, 'netG_tex')
        discriminator = getattr(self, 'netD_tex')
        if self.use_moving_avg:
            g_running = getattr(self, 'g_running')

        gen_embeddings = self.gen_conditions
        rec_embeddings = self.rec_conditions
        orig_embeddings = self.orig_conditions

        ############### multi GPU ###############
        reconst_tex_images, generated_tex_images, cyc_tex_images, orig_id_features,
        orig_age_features, fake_id_features, fake_age_features = \
        generator(gen_in, rec_embeddings, gen_embeddings, orig_embeddings)

        #discriminator pass
        disc_in = generated_tex_images
        disc_out, _, _ = discriminator(disc_in)

        #self-reconstruction loss
        if self.opt.lambda_id_tex > 0:
            loss_G_ID = self.criterionID(reconst_tex_images, gen_in) * self.opt.lambda_id_tex
        else:
            loss_G_ID = torch.zeros(1).cuda()

        #cycle loss
        if self.opt.lambda_cyc_tex > 0:
            loss_G_Cycle = self.criterionCycle(cyc_tex_images, gen_in) * self.opt.lambda_cyc_tex
        else:
            loss_G_Cycle = torch.zeros(1).cuda()

        #content (identity) feature loss
        loss_G_content_reconst = self.content_reconst_criterion(fake_id_features, orig_id_features) * self.opt.lambda_content
        #age feature loss
        loss_G_age_reconst = self.age_reconst_criterion(fake_age_features, gen_embeddings) * self.opt.lambda_age
        #orig age feature loss
        if self.orig_age_features_rec_penalty:
            loss_G_age_reconst += self.age_reconst_criterion(orig_age_features, orig_embeddings) * self.opt.lambda_age

        #GAN loss
        target_classes = torch.cat((self.class_B,self.class_A),0)
        loss_G_GAN = self.criterionGAN(disc_out, target_classes, True, is_gen=True)

        loss_G = (loss_G_GAN + loss_G_ID + loss_G_Cycle + loss_G_content_reconst + \
                  loss_G_age_reconst + loss_G_age_embedding).mean()

        loss_G.backward()
        self.optimizer_G.step()

        if self.use_moving_avg:
            self.accumulate(g_running, generator)

        if infer:
            if self.use_moving_avg:
                with torch.no_grad():
                    orig_id_features_out, _ = g_running.encode(gen_in)
                    #within domain decode
                    if self.opt.lambda_id_tex > 0:
                        if self.use_orig_age_features_within_domain:
                            reconst_tex_images_out = g_running(gen_in, None)
                        else:
                            reconst_tex_images_out, _, _, _ = g_running.decode(orig_id_features_out, None, rec_embeddings)

                    #cross domain decode
                    generated_tex_images_out, _, _, _ = g_running.decode(orig_id_features_out, None, gen_embeddings)
                    #encode generated
                    fake_id_features_out, _ = g_running.encode(generated_tex_images, self.masks)
                    #decode generated
                    if self.opt.lambda_cyc_tex > 0:
                        cyc_tex_images_out, _, _, _ = g_running.decode(fake_id_features_out, None, rec_embeddings)
            else:
                generated_tex_images_out = generated_tex_images
                if self.opt.lambda_id_tex > 0:
                    reconst_tex_images_out = reconst_tex_images
                if self.opt.lambda_cyc_tex > 0:
                    cyc_tex_images_out = cyc_tex_images

        loss_dict = {'loss_G_GAN': loss_G_GAN.mean(), 'loss_G_Cycle': loss_G_Cycle.mean(),
                     'loss_G_ID': loss_G_ID.mean(), 'loss_G_content_reconst': loss_G_content_reconst.mean(),
                     'loss_G_age_reconst': loss_G_age_reconst.mean()}

        return [loss_dict,
                None if not infer else gen_in,
                None if not infer else generated_tex_images_out,
                None if not infer else reconst_tex_images_out,
                None if not infer else cyc_tex_images_out]

    def update_D(self):
        self.optimizer_D.zero_grad()
        self.get_conditions()

        gen_in = self.reals
        generator = getattr(self, 'netG_tex')
        discriminator = getattr(self, 'netD_tex')

        ############### multi GPU ###############
        _, generated_tex_images, _, _, _, _, _, _, _, _, _ = generator(gen_in, None, self.gen_conditions, None, disc_pass=True)

        #fake discriminator pass
        fake_disc_in = generated_tex_images.detach()
        fake_disc_out, _, _ = discriminator(fake_disc_in)

        #real discriminator pass
        real_disc_in = gen_in

        # necessary for R1 regularization
        real_disc_in.requires_grad_()

        real_disc_out, _, _ = discriminator(real_disc_in)

        #Fake GAN loss
        fake_target_classes = torch.cat((self.class_B,self.class_A),0)
        loss_D_fake = self.criterionGAN(fake_disc_out, fake_target_classes, False, is_gen=False)

        #Real GAN loss
        real_target_classes = torch.cat((self.class_A,self.class_B),0)
        loss_D_real = self.criterionGAN(real_disc_out, real_target_classes, True, is_gen=False)

        # R1 regularization (when necessary)
        loss_D_reg = self.R1_reg(real_disc_out, real_disc_in)

        loss_D = (loss_D_fake + loss_D_real + loss_D_reg).mean()
        loss_D.backward()
        self.optimizer_D.step()

        return {'loss_D_real': loss_D_real.mean(), 'loss_D_fake': loss_D_fake.mean(), 'loss_D_reg': loss_D_reg.mean()}

    def inference(self, data):
        # Encode Inputs
        self.encode_input(data, mode='test')
        if self.isEmpty:
            return

        self.numValid = self.valid.sum().item()
        sz = self.reals.size()

        h_tex = sz[2]
        w_tex = sz[3]

        self.fake_B_tex = self.Tensor(self.numClasses, sz[0], sz[1], h_tex, w_tex)
        self.rec_A_tex = self.Tensor(self.numClasses, sz[0], sz[1], h_tex, w_tex)

        with torch.no_grad():
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
                    self.get_conditions(mode='test')

                    if self.gan_mode == 'texture_only' or self.gan_mode == 'seg_only':
                        gen_in = self.reals
                        generator = getattr(self, 'netG_tex')
                        rec_net = getattr(self, 'netG_tex')

                        self.tex_in = gen_in
                        if self.use_orig_age_features_within_domain:
                            within_domain_idx = i
                        else:
                            within_domain_idx = -1

                        if self.isTrain:
                            self.fake_B_tex[i, :, :, :, :] = self.g_running.infer(tex_in, None, self.gen_conditions, within_domain_idx)
                        else:
                            self.fake_B_tex[i, :, :, :, :] = generator.infer(tex_in, None, self.gen_conditions, within_domain_idx)

                        fake_B_tex_rec_input = self.fake_B_tex[i, :, :, :, :]

                        if self.isTrain:
                            self.rec_A_tex[i, :, :, :, :] = self.g_running.infer(fake_B_tex_rec_input, None, self.rec_conditions, within_domain_idx)
                        else:
                            self.rec_A_tex[i, :, :, :, :] = rec_net.infer(fake_B_tex_rec_input, None, self.rec_conditions, within_domain_idx)

            visuals = self.get_visuals()

        return visuals

    def save(self, which_epoch):
        self.save_network(self.netG_tex, 'G_tex', which_epoch, self.gpu_ids)
        self.save_network(self.netD_tex, 'D_tex', which_epoch, self.gpu_ids)
        if self.use_moving_avg:
            self.save_network(self.g_running, 'g_running', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
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

        real_A = util.tensor2im(self.tex_in.data)
        fake_B_tex = util.tensor2im(self.fake_B_tex.data)

        if self.debug_mode:
            rec_A_tex = util.tensor2im(self.rec_A_tex.data[:,:,:,:,:])

        if self.numValid == 1:
            real_A = np.expand_dims(real_A, axis=0)

        for i in range(self.numValid):
            # get the original image and the results for the current samples
            curr_real_A = real_A[i, :, :, :]
            real_A_img = curr_real_A[:, :, :3]

            # set output classes numebr
            if self.traverse:
                out_classes = curr_fake_B_tex.shape[0]
            else:
                out_classes = self.numClasses

            # start with age progression/regression images
            if self.traverse or self.deploy:
                curr_fake_B_tex = fake_B_tex
                orig_dict = OrderedDict([('orig_img', real_A_img)])
            else:
                curr_fake_B_tex = fake_B_tex[:, i, :, :, :]
                orig_dict = OrderedDict([('orig_img_cls_' + str(self.class_A[i].item()), real_A_img)])

            return_dicts[i].update(orig_dict)

            for j in range(out_classes):
                fake_res_tex = curr_fake_B_tex[j, :, :, :3]
                fake_dict_tex = OrderedDict([('tex_trans_to_class_' + str(j), fake_res_tex)])
                return_dicts[i].update(fake_dict_tex)

            if not (self.traverse or self.deploy):
                if self.debug_mode and (not self.fgnet):
                    # continue with tex reconstructions
                    curr_rec_A_tex = rec_A_tex[:, i, :, :, :]
                    orig_dict = OrderedDict([('orig_img2', real_A_img)])
                    return_dicts[i].update(orig_dict)
                    for j in range(self.numClasses):
                        rec_res_tex = curr_rec_A_tex[j, :, :, :3]
                        rec_dict_tex = OrderedDict([('tex_rec_from_class_' + str(j), rec_res_tex)])
                        return_dicts[i].update(rec_dict_tex)

        return return_dicts

class InferenceModel(LATS):
    def forward(self, data):
        return self.inference(data)
