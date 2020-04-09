import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class AdaINModel(BaseModel):
    def name(self):
        return 'AdaIN Model'

    def init_loss_filter(self):
        flags = (True, True, True, True, True, self.flowgan_mode != 'flow_only', self.flowgan_mode != 'flow_only', self.flowgan_mode != 'flow_only',
                 self.flowgan_mode != 'texture_only', self.flowgan_mode != 'texture_only')
        def loss_filter(g_gan, d_real, d_fake, g_cycle, g_id, tex_id_feature_loss, tex_age_feature_loss, tex_age_class_loss, minflow, flowTV):
            return [l for (l,f) in zip((g_gan, d_real, d_fake, g_cycle, g_id, tex_id_feature_loss, tex_age_feature_loss, tex_age_class_loss, minflow, flowTV),flags) if f]
        return loss_filter

    def initialize(self, opt):
        pass

    def set_loader_mode(self):
        if self.use_flow_classes and self.flowgan_mode == 'flow_only':
            return 'uniform_flow'
        else:
            return 'uniform_tex'

    def encode_input(self, data):
        if self.isTrain:
            input_A = data['A']
            input_B = data['B']
            if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                facial_hair_mask_A = data['facial_hair_A']
                facial_hair_mask_B = data['facial_hair_B']
            if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                clothing_items_mask_A = data['clothing_items_A']
                clothing_items_mask_B = data['clothing_items_B']
            if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                neck_mask_A = data['neck_A']
                neck_mask_B = data['neck_B']
            if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
               (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                expanded_A_parsing = data['expanded_A_parsing']
                expanded_B_parsing = data['expanded_B_parsing']

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
                real_A = real_A.cuda()
                real_B = real_B.cuda()
                mask_A = mask_A.cuda()
                mask_B = mask_B.cuda()
                if self.use_parsings:
                    parsing_A = parsing_A.cuda()
                    parsing_B = parsing_B.cuda()
                    if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                        facial_hair_mask_A = facial_hair_mask_A.cuda()
                        facial_hair_mask_B = facial_hair_mask_B.cuda()

                    if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                        clothing_items_mask_A = clothing_items_mask_A.cuda()
                        clothing_items_mask_B = clothing_items_mask_B.cuda()

                    if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                        neck_mask_A = neck_mask_A.cuda()
                        neck_mask_B = neck_mask_B.cuda()

                    if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
                       (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                        expanded_A_parsing = expanded_A_parsing.cuda()
                        expanded_B_parsing = expanded_B_parsing.cuda()

            # rescale masks to [0 1] values
            if self.use_masks:
                mask_A = (mask_A + 1) / 2
                mask_B = (mask_B + 1) / 2

            if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                facial_hair_mask_A = (facial_hair_mask_A + 1) / 2
                facial_hair_mask_B = (facial_hair_mask_B + 1) / 2

            if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                clothing_items_mask_A = (clothing_items_mask_A + 1) / 2
                clothing_items_mask_B = (clothing_items_mask_B + 1) / 2

            if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                neck_mask_A = (neck_mask_A + 1) / 2
                neck_mask_B = (neck_mask_B + 1) / 2

            if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
               (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                expanded_A_parsing = (expanded_A_parsing + 1) / 2
                expanded_B_parsing = (expanded_B_parsing + 1) / 2

            self.reals = torch.cat((real_A, real_B), 0)
            self.masks = torch.cat((mask_A, mask_B), 0)

            if self.use_xy or self.use_rgbxy_flow_inputs:
                bSize, ch, h, w = self.reals.size()
                # create original grid (equivalent to numpy meshgrid)
                x = torch.linspace(-1, 1, steps=w).type_as(self.reals)
                y = torch.linspace(-1, 1, steps=h).type_as(self.reals)

                # pytorch 0.4.1
                xx = x.view(1, -1).repeat(bSize, 1, h, 1)
                yy = y.view(-1, 1).repeat(bSize, 1, 1, w)

                self.xy = torch.cat([xx, yy], 1)

            if self.use_parsings:
                self.parsings = torch.cat((parsing_A, parsing_B), 0)
                if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                    self.facial_hair_masks = torch.cat((facial_hair_mask_A, facial_hair_mask_B), 0)
                else:
                    self.facial_hair_masks = None

                if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                    self.clothing_items_masks = torch.cat((clothing_items_mask_A, clothing_items_mask_B), 0)
                else:
                    self.clothing_items_masks = None

                if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                    self.neck_masks = torch.cat((neck_mask_A, neck_mask_B), 0)
                else:
                    self.neck_masks = None

            else:
                self.parsings = None
                self.facial_hair_masks = None
                self.neck_masks = None

            if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
               (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                self.expanded_parsings = torch.cat((expanded_A_parsing, expanded_B_parsing), 0)
            else:
                self.expanded_parsings = None

            if self.flowgan_mode != 'texture_only' and self.use_landmarks:
                self.landmarks = torch.cat((landmarks_A, landmarks_B), 0)
            else:
                self.landmarks = None

        else:
            inputs = data['Imgs']
            if inputs.dim() > 4:
                inputs = inputs.squeeze(0)

            if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                facial_hair_masks = data['facial_hair']
                if facial_hair_masks.dim() > 4:
                    facial_hair_masks = facial_hair_masks.squeeze(0)

            if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                clothing_items_masks = data['clothing_items']
                if clothing_items_masks.dim() > 4:
                    clothing_items_masks = clothing_items_masks.squeeze(0)

            if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                neck_masks = data['neck']
                if neck_masks.dim() > 4:
                    neck_masks = neck_masks.squeeze(0)

            if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
               (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                expanded_parsings = data['expanded_parsings']

            self.class_A = data['Classes']
            if self.class_A.dim() > 1:
                self.class_A = self.class_A.squeeze(0)

            self.flow_class_A = data['flow_Classes']
            if self.flow_class_A.dim() > 1:
                self.flow_class_A = self.flow_class_A.squeeze(0)

            self.valid = data['Valid'].byte()
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
                if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                    facial_hair_masks = torch.index_select(facial_hair_masks, 0, select_idx)
                if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                    clothing_items_masks = torch.index_select(clothing_items_masks, 0, select_idx)
                if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                    neck_masks = torch.index_select(neck_masks, 0, select_idx)
                if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
                   (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                    expanded_parsings = torch.index_select(expanded_parsings, 0, select_idx)

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
                real_A = real_A.cuda()
                mask_A = mask_A.cuda()
                if self.use_parsings:
                    parsing_A = parsing_A.cuda()
                    if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                        facial_hair_masks = facial_hair_masks.cuda()

                    if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                        clothing_items_masks = clothing_items_masks.cuda()

                    if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                        neck_masks = neck_masks.cuda()

                if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
                   (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                    expanded_parsings = expanded_parsings.cuda()

            # rescale masks to [0 1] values
            if self.use_masks:
                mask_A = (mask_A + 1) / 2

            if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                facial_hair_masks = (facial_hair_masks + 1) / 2

            if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                clothing_items_masks = (clothing_items_masks + 1) / 2

            if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                neck_masks = (neck_masks + 1) / 2

            if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
               (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                expanded_parsings = (expanded_parsings + 1) / 2

            self.reals = real_A
            self.masks = mask_A

            if self.use_xy or self.use_rgbxy_flow_inputs:
                bSize, ch, h, w = self.reals.size()
                # create original grid (equivalent to numpy meshgrid)
                x = torch.linspace(-1, 1, steps=w).type_as(self.reals)
                y = torch.linspace(-1, 1, steps=h).type_as(self.reals)

                # pytorch 0.4.1
                xx = x.view(1, -1).repeat(bSize, 1, h, 1)
                yy = y.view(-1, 1).repeat(bSize, 1, 1, w)

                self.xy = torch.cat([xx, yy], 1)

            if self.use_parsings:
                self.parsings = parsing_A
                if self.flowgan_mode != 'flow_only' and self.no_facial_hair:
                    self.facial_hair_masks = facial_hair_masks
                else:
                    self.facial_hair_masks = None

                if self.flowgan_mode != 'flow_only' and self.no_clothing_items:
                    self.clothing_items_masks = clothing_items_masks
                else:
                    self.clothing_items_masks = None

                if self.flowgan_mode != 'flow_only' and self.no_neck_tex:
                    self.neck_masks = neck_masks
                else:
                    self.neck_masks = None

            else:
                self.parsings = None
                self.facial_hair_masks = None
                self.neck_masks = None

            if (self.flowgan_mode != 'flow_only' and self.use_encoding_net) or \
               (self.flowgan_mode != 'texture_only' and self.use_encoding_net_flow):
                self.expanded_parsings = expanded_parsings
            else:
                self.expanded_parsings = None

        self.masked_reals = self.reals * self.masks

    def generate_random_age_features(self):
        pass

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def discriminate(self, input_images):
        pass

    def forward(self, data, infer=False):
        if self.flowgan_mode == 'flow_only':
            pass
        if self.flowgan_mode == 'texture_only':
            # set classes
            target_classes = torch.cat((self.class_B, self.class_A), 0)
            orig_classes = torch.cat((self.class_A, self.class_B), 0)

            # forward passes
            random_target_age_features = self.generate_random_age_features()
            id_features, age_features = self.netG_tex.encode(self.masked_reals, self.masks)
            reconst_out = self.netG_tex.decode(id_features, age_features)
            fake_out = self.netG_tex.decode(id_features, random_target_age_features)
            fake_id_features, fake_age_features = self.netG_tex.encode(fake_out, self.masks)
            cyc_reconst_out = self.netG_tex.decode(fake_id_features, age_features)
            detached_fake_out = fake_out.detach()

            disc_G_fake = self.discriminate(fake_out, target_classes)
            disc_D_fake = self.discriminate(detached_fake_out, target_classes)
            disc_D_real = self.discriminate(self.masked_reals, orig_classes)

            # calculate generator losses
            tex_age_class_loss = self.kld_loss(age_features, target_classes)
            tex_id_feature_loss = self.recon_criterion(fake_id_features, id_features)
            tex_age_feature_loss = self.recon_criterion(fake_age_features, random_target_age_features)
            tex_identity_reconst_loss = self.recon_criterion(reconst_out, self.masked_reals)
            tex_cyc_reconst_loss = self.recon_criterion(cyc_reconst_out, self.masked_reals)
            g_adv_loss = self.gan_loss(disc_G_fake, True)

            # calculate discriminator losses
            d_fake_loss = self.gan_loss(disc_D_fake, False)
            d_real_loss = self.gan_loss(disc_D_real, True)

        if self.flowgan_mode == 'flow_and_texture':
            pass

    def inference(self, data):
        pass

    def save(self, which_epoch):
        pass

    def update_fixed_params(self):
        pass

    def update_flow_params(self):
        pass

    def update_learning_rate(self):
        pass

    def get_visuals(self):
        pass


class InferenceModel(AdaINModel):
    def forward(self, data):
        return self.inference(data)
