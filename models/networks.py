### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import functools
from torch.autograd import Variable
from torch.autograd import grad as Grad
from torch.autograd import Function
import numpy as np
from math import sqrt
from pdb import set_trace as st

###############################################################################
# Functions
###############################################################################
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.002)
#     elif classname.find('BatchNorm2d') != -1:
#         m.weight.data.normal_(1.0, 0.002)
#         m.bias.data.fill_(0)
#     elif classname.find('GroupNorm') != -1:
#         m.weight.data.normal_(1.0, 0.002)
#         m.bias.data.fill_(0)
#     elif classname.find('LayerNorm') != -1:
#         m.weight.data.normal_(1.0, 0.002)
#         m.bias.data.fill_(0)

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, 16, affine=True)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm, elementwise_affine=True)
    elif norm_type == 'pixel':
        norm_layer = PixelNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG_type, numClasses=2, is_flow=False, n_downsample_global=3,
             n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm='instance',
             gpu_ids=[], cond_length=0, padding_type='reflect', use_cond_resnet_block=False, use_avg_features=False,
             style_dim=100, is_residual=False, res_block_const=1.0, init_type='gaussian', vae_style_encoder=False,
             out_type='rgb', conv_weight_norm=False, decoder_norm='layer', upsample_norm='adain', activation='relu', use_tanh=False,
             truncate_std=False, adaptive_blocks=4, use_resblk_pixel_norm=False, residual_bottleneck=False,
             last_upconv_out_layers=-1, conv_img_kernel_size=1, normalize_mlp=False, modulated_conv=False,
             use_flow_layers=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG_type == 'cond_global':
        netG = CondGlobalGenerator(input_nc, output_nc, ngf, is_flow=is_flow, numClasses=numClasses,
                    n_downsampling=n_downsample_global, n_blocks=n_blocks_global,
                    norm_layer=norm_layer, cond_length=cond_length, padding_type=padding_type,
                    is_residual=is_residual, res_block_const=res_block_const)
    elif netG_type == 'cond_local':
        netG = CondLocalEnhancer(input_nc, output_nc, ngf, is_flow=is_flow, numClasses=numClasses,
                    n_downsample_global=n_downsample_global, n_blocks_global=n_blocks_global,
                    n_local_enhancers=n_local_enhancers, n_blocks_local=n_blocks_local,
                    norm_layer=norm_layer, cond_length=cond_length, padding_type=padding_type,
                    use_cond_resnet_block=use_cond_resnet_block)
    elif netG_type == 'cond_resnet':
        netG = CondResnetGenerator(input_nc, output_nc, ngf=64, is_flow=is_flow, numClasses=numClasses,
                    norm_layer=norm_layer,n_blocks=n_blocks_global, cond_length=cond_length, padding_type=padding_type)
    elif netG_type == 'cond_encoder':
        netG = CondEncoder(input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=norm_layer, numClasses=numClasses,
                     padding_type=padding_type, cond_length=cond_length, use_avg_features=use_avg_features)
    elif 'ada' in netG_type:
        use_style_decoder = 'stylegan_dec' in netG_type
        adaptive_norm = 'adain' if 'adain_gen' in netG_type else 'adaspade'
        netG = AdaINGenerator(input_nc, output_nc, ngf, n_downsampling=n_downsample_global,
                              norm_layer=norm_layer, padding_type=padding_type, style_dim=style_dim,
                              vae_style_encoder=vae_style_encoder, out_type=out_type, adaptive_norm=adaptive_norm,
                              upsample_norm=upsample_norm, conv_weight_norm=conv_weight_norm, decoder_norm=decoder_norm,
                              activation=activation, use_style_decoder=use_style_decoder, use_tanh=use_tanh,
                              truncate_std=truncate_std, adaptive_blocks=adaptive_blocks,
                              use_resblk_pixel_norm=use_resblk_pixel_norm, residual_bottleneck=residual_bottleneck,
                              last_upconv_out_layers=last_upconv_out_layers, conv_img_kernel_size=conv_img_kernel_size,
                              normalize_mlp=normalize_mlp, modulated_conv=modulated_conv, use_flow=use_flow_layers)
    else:
        raise('generator not implemented!')

    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])

    # if (netG_type == 'ada' in netG_type) and conv_weight_norm:
    #     init_type = 'default'
    netG.apply(weights_init(init_type))
    # netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, netD, numClasses=2, n_layers_D=3, norm='instance', use_sigmoid=False,
             num_D=1, num_init_downsample=0, getIntermFeat=False, getFinalFeat=False, use_class_head=False,
             selective_class_loss=False, classify_fakes=False, use_disc_cond_with_class=False,
             mse_class_loss=False, use_norm=True, gpu_ids=[], init_type='gaussian', activation='lrelu'):

    norm_layer = get_norm_layer(norm_type=norm)
    if netD == 'multiscale':
        netD = CondMultiscaleDiscriminator(input_nc, ndf=ndf, numClasses=numClasses,
                        n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                        num_D=num_D, num_init_downsample=num_init_downsample, getIntermFeat=getIntermFeat,
                        getFinalFeat=getFinalFeat, use_class_head=use_class_head,
                        selective_class_loss=selective_class_loss, classify_fakes=classify_fakes,
                        use_disc_cond_with_class=use_disc_cond_with_class, mse_class_loss=mse_class_loss,
                        use_norm=use_norm)
    elif netD == 'perclass':
        netD = PerClassDiscriminator(input_nc, ndf=ndf, numClasses=numClasses,
                        n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                        num_D=num_D, num_init_downsample=num_init_downsample, getIntermFeat=getIntermFeat,
                        getFinalFeat=getFinalFeat, use_class_head=use_class_head,
                        selective_class_loss=selective_class_loss, classify_fakes=classify_fakes,
                        use_disc_cond_with_class=use_disc_cond_with_class, mse_class_loss=mse_class_loss,
                        use_norm=use_norm)
    elif netD == 'aux':
        netD = AuxOutNLayerDiscriminator(input_nc, im_size=256, ndf=ndf, n_layers=6, norm_layer=norm_layer,
                        normalize=False, gpu_ids=[], numClasses=numClasses, classify_fakes=classify_fakes)
    elif netD == 'stylegan':
        netD = StyleGANDiscriminator(input_nc, ndf=ndf, n_layers=6, numClasses=numClasses, actvn=activation,
                                     is_conditional=use_disc_cond_with_class)
    else:
        raise('discriminator not implemented!')

    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])

    # if netD == 'stylegan':
    #     netD.apply(weights_init('default'))
    # else:
    netD.apply(weights_init('gaussian'))
    # netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Data parallel wrapper
##############################################################################
class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            print(name)
            return getattr(self.module, name)

##############################################################################
# Activations
##############################################################################
class BidirectionalLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2, positive_slope=0.1, truncation_point=6.0):
        super(BidirectionalLeakyReLU, self).__init__()
        self.nlrelu = nn.LeakyReLU(negative_slope,True)
        self.plrelu = nn.LeakyReLU(positive_slope,True)
        self.truncation_point = truncation_point

    def forward(self, input):
        out = self.nlrelu(input)
        out = self.truncation_point - self.plrelu(self.truncation_point - out)
        return out

##############################################################################
# Losses
##############################################################################
# KL Divergence loss used in VAE with an image encoder
class VAEKLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

class GaussianKLDLoss(nn.Module):
    def __init__(self):
        super(GaussianKLDLoss, self).__init__()
        self.eps = 1e-8

    def __call__(self, input, target_mean, target_sigma):
        input_mean = input.mean(dim=1)
        input_sigma = input.std(dim=1)

        # treat target as mu1, sigma1 and input as mu2, sigma2
        loss = torch.mean(torch.log(input_sigma / target_sigma) + \
                          0.5 * (-1 + ((target_sigma**2 + ((target_mean - input_mean)**2)) / (input_sigma ** 2 + self.eps))))

        # # treat input as mu1, sigma1 and target as mu2, sigma2
        # loss = torch.mean(torch.log(target_sigma / input_sigma) + \
        #                   0.5 * (-1 + ((input_sigma**2 + ((input_mean - target_mean)**2)) / (target_sigma ** 2))))
        return loss


class FeatureConsistency(nn.Module):
    def __init__(self):
        super(FeatureConsistency, self).__init__()

    def __call__(self,input,target):
        return torch.mean(torch.abs(input - target))


# Defines the gadient penalty for WGAN-GP loss
# Only supported on single GPU
class gradPenalty(nn.Module):
    def __init__(self, lambda_gp=10.0):
        super(gradPenalty, self).__init__()
        self.lambda_gp = lambda_gp

    def __call__(self, d_out, d_in):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(d_out.size()).cuda()
        dydx = torch.autograd.grad(outputs=d_out,
                                   inputs=d_in,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2) * self.lambda_gp

class R1_reg(nn.Module):
    def __init__(self, lambda_r1=10.0):
        super(R1_reg, self).__init__()
        self.lambda_r1 = lambda_r1

    def __call__(self, d_out, d_in):
        """Compute gradient penalty: (L2_norm(dy/dx))**2."""
        b = d_in.shape[0]
        if isinstance(d_out[0], list):
            r1_reg = 0
            for d_out_i in d_out:
                pred = d_out_i[-1]
                dydx = torch.autograd.grad(outputs=pred.mean(),
                                           inputs=d_in,
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
                dydx_sq = dydx.pow(2)
                assert (dydx_sq.size() == d_in.size())
                r1_reg += dydx_sq.sum() / b
        else:
            dydx = torch.autograd.grad(outputs=d_out.mean(),
                                       inputs=d_in,
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            dydx_sq = dydx.pow(2)
            assert (dydx_sq.size() == d_in.size())
            r1_reg = dydx_sq.sum() / b

        return r1_reg * self.lambda_r1

class WGANLoss(nn.Module):
    def __init__(self):
        pass
    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                if target_is_real:
                    loss += -pred.mean()
                else:
                    loss += pred.mean()
            return loss
        else:
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                # pytorch 0.3.1
                # real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                # pytorch 0.4.1
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                # pytorch 0.3.1
                # fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                # pytorch 0.4.1
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class SelectiveClassesHingeGANLoss(nn.Module):
    def __init__(self):
        super(SelectiveClassesHingeGANLoss, self).__init__()
        self.relu = nn.ReLU()

    def __call__(self, input, target_classes, target_is_real, is_gen=False):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                bSize = pred.shape[0]
                b_ind = torch.arange(bSize).long()
                relevant_inputs = pred[b_ind, target_classes, :, :]
                if is_gen:
                    loss += -relevant_inputs.mean()
                    # loss += self.relu(1-relevant_inputs).mean()
                elif target_is_real:
                    loss += self.relu(1-relevant_inputs).mean()
                else:
                    loss += self.relu(1+relevant_inputs).mean()
        else:
            bSize = input.shape[0]
            b_ind = torch.arange(bSize).long()
            relevant_inputs = input[b_ind, target_classes, :, :]
            if is_gen:
                loss = -relevant_inputs.mean()
                # loss = self.relu(1-relevant_inputs).mean()
            elif target_is_real:
                loss = self.relu(1-relevant_inputs).mean()
            else:
                loss = self.relu(1+relevant_inputs).mean()

        return loss
class NonSatGANLoss(nn.Module):
    def __init__(self):
        super(NonSatGANLoss, self).__init__()
        self.sofplus = nn.Softplus()

    def __call__(self, input, target_classes, target_is_real, is_gen=False):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                if target_is_real:
                    loss += self.sofplus(-pred).mean()
                else:
                    loss += self.sofplus(pred).mean()
        else:
            if target_is_real:
                loss = self.sofplus(-input).mean()
            else:
                loss = self.sofplus(input).mean()

        return loss

class SelectiveClassesNonSatGANLoss(nn.Module):
    def __init__(self):
        super(SelectiveClassesNonSatGANLoss, self).__init__()
        self.sofplus = nn.Softplus()

    def __call__(self, input, target_classes, target_is_real, is_gen=False):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                bSize = pred.shape[0]
                b_ind = torch.arange(bSize).long()
                relevant_inputs = pred[b_ind, target_classes, :, :]
                if target_is_real:
                    loss += self.sofplus(-relevant_inputs).mean()
                else:
                    loss += self.sofplus(relevant_inputs).mean()
        else:
            bSize = input.shape[0]
            b_ind = torch.arange(bSize).long()
            relevant_inputs = input[b_ind, target_classes, :, :]
            if target_is_real:
                loss = self.sofplus(-relevant_inputs).mean()
            else:
                loss = self.sofplus(relevant_inputs).mean()

        return loss

class SelectiveClassesLSGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(SelectiveClassesLSGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_classes, target_is_real, is_gen=False):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                bSize = pred.shape[0]
                b_ind = torch.arange(bSize).long()
                relevant_inputs = pred[b_ind, target_classes, :, :]
                target_tensor = self.get_target_tensor(relevant_inputs, target_is_real)
                loss += self.loss(relevant_inputs, target_tensor)
            return loss
        else:
            bSize = input.shape[0]
            b_ind = torch.arange(bSize).long()
            relevant_inputs = input[b_ind, target_classes, :, :]
            target_tensor = self.get_target_tensor(relevant_inputs[-1], target_is_real)
            return self.loss(relevant_inputs, target_tensor)


class FeatureTripletLoss(nn.Module):
    def __init__(self):
        super(FeatureTripletLoss, self).__init__()
        self.loss = nn.TripletMarginLoss()

    def __call__(self, anchor, positive, negative):
        if isinstance(anchor, list):
            loss = 0
            bSize = []
            numel = []
            for i in range(len(anchor)):
                b = anchor[i].shape[0]
                bSize += [b]
                numel += [anc.numel() / b]

            minel = min(numel)
            weights = [minel / n for n in numel]

            for i in range(len(anchor)):
                anc = anchor[i]
                pos = positive[i]
                neg = negative[i]
                a = anc.view(bSize[i],-1)
                p = pos.view(bSize[i],-1)
                n = neg.view(bSize[i],-1)
                loss += weights[i] * self.loss(a,p,n)

        else:
            bSize = anchor.shape[0]
            a = anchor.view(bSize,-1)
            p = positive.view(bSize,-1)
            n = negative.view(bSize,-1)
            loss = self.loss(a,p,n)

        return loss


class AuxLoss(nn.Module):
    def __init__(self, multiple_atts=False, mse_loss=False, num_classes=2, class_loss='cross_entropy'):#, is_dual=False):
        super(AuxLoss, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = mse_loss
        if mse_loss:
            self.loss = nn.MSELoss()
            self.is_bce = False
        elif multiple_atts:
            self.loss = nn.BCEWithLogitsLoss()
            self.is_bce = True
        else:
            self.is_bce = False
            if class_loss == 'multi_label_margin':
                self.loss = nn.MultiLabelMarginLoss()
            elif class_loss == 'multi_label_soft_margin':
                self.loss = nn.MultiLabelSoftMarginLoss()
            elif class_loss == 'multi_margin':
                self.loss = nn.MultiMarginLoss()
            else:
                self.loss = nn.CrossEntropyLoss() # cross entropy loss

    def one_hot(self, target):
        """Convert label indices to one-hot vector"""
        batch_size = target.size(0)
        out = torch.zeros(batch_size, self.num_classes).cuda()
        out[range(batch_size), target.long()] = 1

        return out

    def __call__(self, input, target):
        if isinstance(input, list):
            loss = 0
            for i in range(len(input)):
                # inp = input[i].view(-1, self.num_classes)
                # target = target.expand(input[i].size(1),input[i].size(0)).transpose(1,0).contiguous().view(-1)
                if self.mse_loss:
                    inp = input[i].view(-1, 1)
                    target = target.float()
                else:
                    inp = input[i].view(-1, self.num_classes)

                if self.is_bce:
                    targets = self.one_hot(target)
                    loss += self.loss(inp, targets, size_average=False)/ input[i].size(0)
                else:
                    loss += self.loss(inp, target)

            loss = loss / len(input)
        else:
            # inp = input.view(-1, self.num_classes)
            # target = target.expand(input.size(1),input.size(0)).transpose(1,0).contiguous().view(-1)
            if self.mse_loss:
                inp = input.view(-1, 1)
                target = target.float()
            else:
                inp = input.view(-1, self.num_classes)

            if self.is_bce:
                targets = self.one_hot(target)
                loss = self.loss(inp, targets, size_average=False)/ input.size(0)
            else:
                loss = self.loss(inp, target)

        return loss

class LandmarkLoss(nn.Module):
    def __init__(self, avgs, stds):
        super(LandmarkLoss, self).__init__()
        self.avgs = avgs.cuda() # should be [num_classes, 106, 2]
        self.stds = stds.cuda() # should be [num_classes, 106, 1]
        self.num_landmarks = self.avgs.shape[1]
        self.dist = nn.PairwiseDistance()
        self.zeros = torch.zeros(1,self.num_landmarks).cuda()

    def forward(self, input, target):
        avgs = self.avgs[target,:,:]
        stds = self.stds[target,:]
        dists = self.dist(input.view(-1,2), avgs.view(-1,2)).view(-1, self.num_landmarks)
        loss = torch.max((dists - stds * 1.5), self.zeros).mean()

        return loss

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids, valid_layers='all'):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
        self.register_buffer('vgg_mean', self.mean)
        self.register_buffer('vgg_std', self.std)
        if valid_layers == 'all':
            self.weights = [1.0, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        elif valid_layers == 'last':
            self.weights = [0.0, 0.0, 0.0, 1.0/8, 1.0/4, 1.0]
        else:
            self.weights = [0.25, 0.5, 1.0, 0.0, 0.0, 0.0]

    def vgg_preprocess(self, x):
        x = (x + 1) / 2
        x = self.vgg_std * (x - self.vgg_mean)
        return x

    def forward(self, x, y):
        x_vgg, y_vgg = [x] + self.vgg(self.vgg_preprocess(x)), [y] + self.vgg(self.vgg_preprocess(y))
        loss = 0

        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class NormalizedVGGLoss(nn.Module):
    def __init__(self, gpu_ids, valid_layers='all'):
        super(NormalizedVGGLoss, self).__init__()
        self.vgg = Vgg16().cuda().eval()
        self.norm = nn.InstanceNorm2d(512, affine=False)
        self.criterion = nn.L1Loss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
        self.register_buffer('vgg_mean', self.mean)
        self.register_buffer('vgg_std', self.std)

    def vgg_preprocess(self, x):
        x = (x + 1) / 2
        x = self.vgg_std * (x - self.vgg_mean)
        return x

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(self.vgg_preprocess(x)), self.vgg(self.vgg_preprocess(y))
        norm_x_vgg, norm_y_vgg = self.norm(x_vgg), self.norm(y_vgg)
        loss = self.criterion(norm_x_vgg, norm_y_vgg.detach())
        return loss

# Defines the minimal flow loss
class MinFlowLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor, norm='l1'):
        super(MinFlowLoss, self).__init__()
        self.tensor = tensor
        self.norm = norm
        if self.norm == 'l1':
            self.loss = nn.L1Loss()
        elif self.norm == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = None

    def __call__(self, input):
        bSize, ch, h, w = input.size()

        assert ch == 2, "input grid must have 2 channels"
        # create original grid (equivalent to numpy meshgrid)
        x = torch.linspace(-1, 1, steps=w).type_as(self.tensor(0))
        y = torch.linspace(-1, 1, steps=h).type_as(self.tensor(0))

        # pytorch 0.4.1
        xx = x.view(1, -1).repeat(bSize, 1, h, 1)
        yy = y.view(-1, 1).repeat(bSize, 1, 1, w)

        orig_grid = torch.cat([xx, yy], 1)

        #calculate loss
        if  self.norm == 'l1' or self.loss == 'mse':
            flow_loss = self.loss(input,orig_grid)
        else:
            flow_loss = (input - orig_grid).norm(p=2,dim=1).mean()

        return flow_loss

# Defines the flow total variation loss
class FlowTVLoss(nn.Module):
    def __init__(self, isotropic=False, tensor=torch.FloatTensor):
        super(FlowTVLoss, self).__init__()
        self.isotropic = isotropic
        self.tensor = tensor

    def __call__(self, input):
        bSize, ch, h, w = input.size()

        # create original grid (equivalent to numpy meshgrid)
        x = torch.linspace(-1, 1, steps=w).type_as(self.tensor(0))
        y = torch.linspace(-1, 1, steps=h).type_as(self.tensor(0))

        # pytorch 0.3.1
        # xx = Variable(x.view(1, -1).repeat(bSize, 1, h, 1), requires_grad=False)
        # yy = Variable(y.view(-1, 1).repeat(bSize, 1, 1, w), requires_grad=False)

        # pytorch 0.4.1
        xx = x.view(1, -1).repeat(bSize, 1, h, 1)
        yy = y.view(-1, 1).repeat(bSize, 1, 1, w)

        orig_grid = torch.cat([xx, yy], 1)

        # get flow field
        flow = input - orig_grid

        # calculate first derivatives
        # pytorch 0.3.1
        # x_bound = Variable(self.tensor(bSize, ch, h, 1).fill_(0))
        # y_bound = Variable(self.tensor(bSize, ch, 1, w).fill_(0))
        # pytorch 0.4.1
        x_bound = self.tensor(bSize, ch, h, 1).fill_(0)
        y_bound = self.tensor(bSize, ch, 1, w).fill_(0)
        Dx = flow[:,:,:,:w-1] - flow[:,:,:,1:w]
        Dy = flow[:,:,:h-1,:] - flow[:,:,1:h,:]
        Dx = torch.cat((Dx, x_bound), 3)
        Dy = torch.cat((Dy, y_bound), 2)

        # find total variation
        if self.isotropic:
            grad = torch.cat((Dx, Dy), 1)
            totalVariation = grad.norm(p=2, dim=1).sum(2).sum(1).mean()
        else:
            totalVariation = (torch.abs(Dx) + torch.abs(Dy)).sum(3).sum(2).sum(1).mean()

        return totalVariation


##############################################################################
# Embedding
##############################################################################
#gaussian filter was taken from https://github.com/ducha-aiki/ucn-pytorch/blob/master/Utils.py
class GaussianFilter(nn.Module):
    def __init__(self, embedding_dim, kernlen=9, sigma=1.6):
        super(GaussianFilter, self).__init__()
        self.embedding_dim = embedding_dim
        weight = self.calculate_weights(kernlen=kernlen, sigma=sigma).expand(self.embedding_dim,-1,-1,-1)
        self.register_buffer('buf', weight)
        self.buf = self.buf
        return

    def circularGaussKernel(self, kernlen=None, circ_zeros=False, sigma=None, norm=False):
        assert ((kernlen is not None) or sigma is not None)
        if kernlen is None:
            kernlen = int(2.0 * 3.0 * sigma + 1.0)
            if (kernlen % 2 == 0):
                kernlen = kernlen + 1;
            halfSize = kernlen / 2;
        halfSize = kernlen / 2;
        self.halfSize = int(halfSize)
        r2 = float(halfSize*halfSize)
        if sigma is None:
            sigma2 = 0.9 * r2;
            sigma = np.sqrt(sigma2)
        else:
            sigma2 = 2.0 * sigma * sigma
        x = np.linspace(-halfSize,halfSize,kernlen)
        xv, yv = np.meshgrid(x, x, sparse=False, indexing='xy')
        distsq = (xv)**2 + (yv)**2
        kernel = np.exp(-( distsq/ (sigma2)))
        if circ_zeros:
            kernel *= (distsq <= r2).astype(np.float32)
        if norm:
            kernel /= np.sum(kernel)
        return kernel

    def calculate_weights(self, kernlen=None, sigma=None):
        assert ((kernlen is not None) or sigma is not None)
        kernel = self.circularGaussKernel(kernlen=kernlen, sigma=sigma)
        h,w = kernel.shape
        return torch.from_numpy(kernel.astype(np.float32)).view(1,1,h,w);

    def forward(self, x):
        return F.conv2d(x, self.buf, padding=int(self.halfSize), groups=self.embedding_dim)


class LandmarksEmbedding(nn.Module):
    def __init__(self, num_landmarks=106, embedding_dim=5):
        super(LandmarksEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_landmarks+1, embedding_dim, padding_idx=0, max_norm=np.sqrt(embedding_dim), norm_type=2).cuda()
        self.gaussian = GaussianFilter(embedding_dim, kernlen=9, sigma=1.6).cuda()

    def forward(self, input):
        embeddings = self.embedding(input)
        embeddings = embeddings.permute(0,1,4,2,3).squeeze(1)
        output = self.gaussian(embeddings)
        return output


##############################################################################
# Generator
##############################################################################
class GridSparseSampler(nn.Module):
    def __init__(self, im_size=256, num_landmarks=106):
        super(GridSparseSampler, self).__init__()
        self.size = im_size
        self.num_landmarks = num_landmarks

    def forward(self, input, grid):
        bSize = input.shape[0]
        normalized_input = (-1 + (2 / (self.size - 1)) * input).view(-1,1,1,self.num_landmarks,2)
        dists = torch.sqrt(torch.sum((normalized_input - grid.view(-1,self.size,self.size,1,2))**2,dim=4))
        min_dists_idxs = torch.argmin(dists.view(-1, self.size*self.size, self.num_landmarks),dim=1)
        rows = min_dists_idxs / self.size
        cols = min_dists_idxs - rows * self.size
        min_dists_locations = torch.cat((rows.view(-1,self.num_landmarks,1), cols.view(-1,self.num_landmarks,1)), 2)
        normalized_out = grid[torch.arange(bSize).view(1,bSize).repeat(106,1).flatten(), min_dists_locations[:,:,0].flatten(),
                             min_dists_locations[:,:,1].flatten(),:].view(bSize,self.num_landmarks,2)
        out = (normalized_out + 1) * ((self.size - 1) / 2)

        return out

class CondLocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, is_flow=False, numClasses=2, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect',
                 cond_length=(0,0), use_cond_resnet_block=False):
        super(CondLocalEnhancer, self).__init__()
        self.is_flow = is_flow
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if self.is_flow:
            output_nc = 2
            self.sampler = nn.functional.grid_sample

        self.n_local_enhancers = n_local_enhancers
        self.use_cond_resnet_block = use_cond_resnet_block
        cond_global_length, cond_local_length = cond_length[0], cond_length[1]
        ###### global generator model #####
        ngf_global = ngf #* (2**n_local_enhancers)
        global_model = CondGlobalGenerator(input_nc, output_nc, ngf=ngf_global, is_flow=self.is_flow, numClasses=numClasses,
                                           n_downsampling=n_downsample_global, n_blocks=n_blocks_global,
                                           norm_layer=norm_layer, cond_length=cond_global_length, padding_type=padding_type)
        self.encoder = global_model.encoder
        model_global_decoder = global_model.decoder
        model_global_decoder = [model_global_decoder[i] for i in range(len(model_global_decoder)-3)] # get rid of final convolution layers
        self.decoder = nn.Sequential(*model_global_decoder)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            # ngf_global = ngf * (2**(n_local_enhancers-n))
            ngf_global = int(ngf * (2**(n_local_enhancers-n-1)))
            model_downsample = [padding_layer(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_residual = []
            if self.use_cond_resnet_block:
                model_residual += [CondResnetBlock(ngf_global * 2, cond_local_length, padding_type=padding_type, norm_layer=norm_layer)]

            for i in range(n_blocks_local):
                model_residual += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            if self.use_cond_resnet_block:
                model_upsample = [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  norm_layer(ngf_global), nn.ReLU(True)]
            else:
                model_upsample = [nn.ConvTranspose2d(ngf_global * 2 + cond_local_length, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                # model_upsample += [padding_layer(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
                model_upsample += [padding_layer(3), nn.Conv2d(ngf_global, output_nc, kernel_size=7, padding=0), nn.Tanh()]

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_residual))
            setattr(self, 'model'+str(n)+'_3', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, cond, parsing=None, mask=None, facial_hair_mask=None,
                clothing_items_mask=None, neck_mask=None, expanded_parsings=None,
                landmarks=None, features=None):

        ### create input pyramid
        cond_global, cond_local = cond[0], cond[1]

        if self.is_flow:
            if features is not None:
                input_downsampled = [torch.cat((expanded_parsings, features),1)]
            elif parsing is None and landmarks is None:
                input_downsampled = [input]
            elif landmarks is None:
                input_downsampled = [parsing]
            else:
                input_downsampled = [torch.cat((parsing, landmarks),1)]

            for i in range(self.n_local_enhancers):
                input_downsampled.append(self.downsample(input_downsampled[-1]))

            ### output at coarest level
            output_prev_enc_out = self.encoder(input_downsampled[-1])
            output_prev_dec_in = torch.cat((output_prev_enc_out, cond_global), 1)
            flow_prev = self.decoder(output_prev_dec_in)
            ### build up one layer at a time
            for n_local_enhancers in range(1, self.n_local_enhancers+1):
                model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
                model_residual = getattr(self, 'model'+str(n_local_enhancers)+'_2')
                model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_3')
                input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
                if self.use_cond_resnet_block:
                    residual_model_in = torch.cat((model_downsample(input_i) + flow_prev, cond_local), 1)
                else:
                    residual_model_in = model_downsample(input_i) + flow_prev

                residual_model_out = model_residual(residual_model_in)
                if self.use_cond_resnet_block:
                     upsample_model_in = residual_model_out
                else:
                    upsample_model_in = torch.cat((residual_model_out, cond_local), 1)

                flow_prev = model_upsample(upsample_model_in)

            flow = flow_prev.permute(0, 2, 3, 1)
            if parsing is not None:
                parsing_out = self.sampler(parsing, flow, padding_mode='zeros')
            else:
                parsing_out = None

            if landmarks is not None:
                landmarks_out = self.sampler(landmarks, flow, padding_mode='zeros')
            else:
                landmarks_out = None

            if mask is not None:
                warped_mask = self.sampler(mask, flow, padding_mode='zeros')
            else:
                warped_mask = None

            if facial_hair_mask is not None:
                warped_facial_hair_mask = self.sampler(facial_hair_mask, flow, padding_mode='zeros')
            else:
                warped_facial_hair_mask = None

            if clothing_items_mask is not None:
                warped_clothing_items_mask = self.sampler(clothing_items_mask, flow, padding_mode='zeros')
            else:
                warped_clothing_items_mask = None

            if neck_mask is not None:
                warped_neck_mask = self.sampler(neck_mask, flow, padding_mode='zeros')
            else:
                warped_neck_mask = None

            if expanded_parsings is not None:
                warped_expanded_parsings = self.sampler(expanded_parsings, flow, mode='nearest')
            else:
                warped_expanded_parsings = None

            out = self.sampler(input, flow, padding_mode='zeros')
            if mask is not None:
                masked_out = out * warped_mask
            else:
                masked_out = None

            return out, masked_out, flow.permute(0, 3, 1, 2), warped_mask, parsing_out, warped_facial_hair_mask, \
                   warped_clothing_items_mask, warped_neck_mask, warped_expanded_parsings, landmarks_out
        else:
            input_downsampled = [input]
            for i in range(self.n_local_enhancers):
                input_downsampled.append(self.downsample(input_downsampled[-1]))

            ### output at coarest level
            output_prev_enc_out = self.encoder(input_downsampled[-1])
            output_prev_dec_in = torch.cat((output_prev_enc_out, cond_global), 1)
            output_prev = self.decoder(output_prev_dec_in)
            ### build up one layer at a time
            for n_local_enhancers in range(1, self.n_local_enhancers+1):
                model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
                model_residual = getattr(self, 'model'+str(n_local_enhancers)+'_2')
                model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_3')
                input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
                if self.use_cond_resnet_block:
                    residual_model_in = torch.cat((model_downsample(input_i) + output_prev, cond_local), 1)
                else:
                    residual_model_in = model_downsample(input_i) + output_prev

                residual_model_out = model_residual(residual_model_in)
                if self.use_cond_resnet_block:
                     upsample_model_in = residual_model_out
                else:
                    upsample_model_in = torch.cat((residual_model_out, cond_local), 1)

                output_prev = model_upsample(upsample_model_in)
            return output_prev

class CondGlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, is_flow=False, numClasses=2,
                 n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', cond_length=0, is_residual=False, res_block_const=1.0):
        assert(n_blocks >= 0)
        super(CondGlobalGenerator, self).__init__()
        self.is_flow = is_flow
        self.is_residual = is_residual

        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if self.is_flow:
            output_nc = 2
            self.sampler = nn.functional.grid_sample
            # self.sparse_sampler = GridSparseSampler()

        activation = nn.ReLU(True)

        encoder = [padding_layer(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, res_block_const=res_block_const)]

        self.encoder = nn.Sequential(*encoder)

        ### upsample
        decoder = [nn.ConvTranspose2d(ngf * mult + cond_length, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                    norm_layer(int(ngf * mult / 2)), activation]

        for i in range(1, n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]

        decoder += [padding_layer(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if not self.is_residual:
            decoder += [nn.Tanh()]

        self.decoder = nn.Sequential(*decoder)

    def forward(self, input, cond, parsing=None, mask=None, facial_hair_mask=None, clothing_items_mask=None,
                neck_mask=None, expanded_parsings=None, landmarks=None, features=None, xy=None,
                use_rgbxy_inputs=False, use_xy_inputs=False):
        if self.is_flow:
            if features is not None:
                enc_in = torch.cat((expanded_parsings, features),1)
            elif parsing is None and landmarks is None:
                enc_in = input
            elif landmarks is None or (landmarks.shape != parsing.shape):
                enc_in = parsing
            else:
                enc_in = torch.cat((parsing, landmarks),1)

            if use_rgbxy_inputs:
                if parsing is None:
                    enc_in = torch.cat((enc_in, xy),1)
                else:
                    enc_in = torch.cat((enc_in, input, xy),1)
            elif use_xy_inputs:
                enc_in = torch.cat((enc_in, xy),1)

            enc_out = self.encoder(enc_in)
            dec_in = torch.cat((enc_out, cond), 1)
            flow = self.decoder(dec_in)
            flow = flow.permute(0, 2, 3, 1)
            if parsing is not None:
                parsing_out = self.sampler(parsing, flow, padding_mode='zeros', mode='nearest')
            else:
                parsing_out = None

            if landmarks is not None and (landmarks.shape == parsing.shape):
                landmarks_out = self.sampler(landmarks, flow, padding_mode='zeros')
            elif landmarks is not None:
                landmarks_out = self.sparse_sampler(landmarks, flow)
            else:
                landmarks_out = None

            if xy is not None:
                warped_xy = self.sampler(xy, flow, padding_mode='border')
            else:
                warped_xy = None

            if mask is not None:
                warped_mask = self.sampler(mask, flow, padding_mode='zeros', mode='nearest')
            else:
                warped_mask = None

            if facial_hair_mask is not None:
                warped_facial_hair_mask = self.sampler(facial_hair_mask, flow, padding_mode='zeros', mode='nearest')
            else:
                warped_facial_hair_mask = None

            if clothing_items_mask is not None:
                warped_clothing_items_mask = self.sampler(clothing_items_mask, flow, padding_mode='zeros', mode='nearest')
            else:
                warped_clothing_items_mask = None

            if neck_mask is not None:
                warped_neck_mask = self.sampler(neck_mask, flow, padding_mode='zeros', mode='nearest')
            else:
                warped_neck_mask = None

            if expanded_parsings is not None:
                warped_expanded_parsings = self.sampler(expanded_parsings, flow, padding_mode='zeros', mode='nearest')
            else:
                warped_expanded_parsings = None

            out = self.sampler(input, flow, padding_mode='zeros')
            if mask is not None:
                masked_out = out * warped_mask
            else:
                masked_out = None

            return out, masked_out, flow.permute(0, 3, 1, 2), warped_mask, parsing_out, warped_facial_hair_mask, \
                   warped_clothing_items_mask, warped_neck_mask, warped_expanded_parsings, landmarks_out, warped_xy
        else:
            enc_out = self.encoder(input)
            dec_in = torch.cat((enc_out, cond), 1)
            dec_out = self.decoder(dec_in)
            if self.is_residual:
                out = torch.tanh(dec_out + input)
            else:
                out = dec_out

            return out


##############################################################################
# AdaIN Generator
##############################################################################
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)

class ModulatedConv2d(nn.Module):
    def __init__(self, fin, fout, kernel_size, padding_type='reflect', upsample=False, downsample=False, latent_dim=256, normalize_mlp=False):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = fin
        self.out_channels = fout
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample
        padding_size = kernel_size // 2
        if kernel_size == 1:
            self.demudulate = False
        else:
            self.demudulate = True

        self.weight = nn.Parameter(torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, fout, 1, 1))
        self.conv = F.conv2d

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(EqualLinear(latent_dim, fin), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(latent_dim, fin)

        self.blur = Blur(fout)

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding_size)
        else:
            self.padding = nn.ZeroPad2d(padding_size)

        if self.upsample:
            self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        if self.downsample:
            self.downsampler = nn.AvgPool2d(2)

        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input, latent):
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        s = 1 + self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight
        if self.demudulate:
            d = torch.rsqrt((weight ** 2).sum(4).sum(3).sum(2) + 1e-5).view(-1, self.out_channels, 1, 1, 1)
            weight = (d * weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = self.upsampler(input)

        if self.downsample:
            input = self.blur(input)

        b,_,h,w = input.shape
        input = input.view(1,-1,h,w)
        input = self.padding(input)
        out = self.conv(input, weight, groups=b).view(b, self.out_channels, h, w) + self.bias

        if self.downsample:
            out = self.downsampler(out)

        if self.upsample:
            out = self.blur(out)

        return out

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)

class SpatialSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpatialSelfAttention, self).__init__()
        self.query_conv = nn.utils.spectral_norm(nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0))
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0))
        self.value_conv = nn.utils.spectral_norm(nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0))
        self.output_conv = nn.utils.spectral_norm(nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0))
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        b, c, h, w = input.shape
        num_locations = h*w
        queries = self.query_conv(input).view(b,c,num_locations).permute(0,2,1)
        keys = self.key_conv(input).view(b,c,num_locations)

        attn = self.softmax(torch.bmm(queries,keys))

        values = self.value_conv(input).view(b, c, num_locations)
        attended_in = torch.bmm(values, attn.permute(0,2,1)).view(b,c,h,w)
        out = self.output_conv(attn) * self.gamma + input

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.weight.view(*shape) + self.bias.view(*shape)
        return x

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, padding_type='reflect'):
        super(SPADE, self).__init__()
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            padding_layer(1),
            nn.Conv2d(label_nc, nhidden, kernel_size=3),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(padding_layer(1), nn.Conv2d(nhidden, norm_nc, kernel_size=3))
        self.mlp_beta = nn.Sequential(padding_layer(1), nn.Conv2d(nhidden, norm_nc, kernel_size=3))

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class AdaSPADE(nn.Module):
    def __init__(self, norm_nc, label_nc=3, latent_dim=256, nhidden=64, padding_type='reflect',
                 conv_weight_norm=False, actvn='relu'):
        super(AdaSPADE, self).__init__()
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
            linear = EqualLinear
        else:
            conv2d = nn.Conv2d
            linear = nn.Linear

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2,True)
        elif actvn == 'blrelu':
            activation = BidirectionalLeakyReLU()
        else:
            activation = nn.ReLU(True)

        self.num_channels = norm_nc
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128

        self.mlp_shared = nn.Sequential(
            padding_layer(1),
            conv2d(label_nc, nhidden, kernel_size=3),
            activation
        )
        self.mlp_label_gamma = nn.Sequential(padding_layer(1), conv2d(nhidden, norm_nc, kernel_size=3))
        self.mlp_label_beta = nn.Sequential(padding_layer(1), conv2d(nhidden, norm_nc, kernel_size=3))
        # self.mlp_class_std = linear(latent_dim, norm_nc)
        # self.mlp_class_mean = linear(latent_dim, norm_nc)
        # if conv_weight_norm:
        #     self.mlp_class_std.linear.bias.data = 1
        #     self.mlp_class_mean.linear.bias.data = 0
        self.class_std = None
        self.class_mean = None

    def forward(self, x, segmap, latent=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map and latent code
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        label_std = self.mlp_label_gamma(actv)
        label_mean = self.mlp_label_beta(actv)
        class_std = self.mlp_class_std(latent).view(-1,self.num_channels,1,1)
        class_mean = self.mlp_class_mean(latent).view(-1,self.num_channels,1,1)

        # apply scale and bias
        out = normalized * self.class_std.view(-1, self.num_channels, 1, 1) * (1 + label_std) + \
              self.class_mean.view(-1, self.num_channels, 1, 1) + label_mean
        # out = normalized * class_std * (1 + label_std) + class_mean + label_mean

        return out

class AdaIN2d(nn.Module):
    # this is the StyleGAN paper implementation
    def __init__(self, norm_nc, latent_dim=256, conv_weight_norm=False, truncate_std=False, normalize_affine_output=False):
        super(AdaIN2d, self).__init__()
        if conv_weight_norm:
            linear = EqualLinear
        else:
            linear = nn.Linear

        activation = BidirectionalLeakyReLU(0,0.1,1.0)
        self.num_channels = norm_nc
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        if truncate_std:
            self.mlp_class_std = nn.Sequential(linear(latent_dim, norm_nc), activation)
            self.mlp_class_mean = linear(latent_dim, norm_nc)
        elif normalize_affine_output:
            self.mlp_class_std = nn.Sequential(linear(latent_dim, norm_nc), PixelNorm())
            self.mlp_class_mean = nn.Sequential(linear(latent_dim, norm_nc), PixelNorm())
        else:
            self.mlp_class_std = linear(latent_dim, norm_nc)
            self.mlp_class_mean = linear(latent_dim, norm_nc)

        if conv_weight_norm:
            if truncate_std or normalize_affine_output:
                self.mlp_class_std[0].linear.bias.data.fill_(1)
                self.mlp_class_mean[0].linear.bias.data.zero_()
            else:
                self.mlp_class_std.linear.bias.data.fill_(1)
                self.mlp_class_mean.linear.bias.data.zero_()

    def forward(self, x, seg=None, latent=None):
        # seg is not used here it only exists to match signatures with adaspade

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on latent code
        class_std = self.mlp_class_std(latent).view(-1,self.num_channels,1,1)
        class_mean = self.mlp_class_mean(latent).view(-1,self.num_channels,1,1)

        # apply scale and bias
        out = normalized * class_std + class_mean

        return out


class AdaptiveInstanceNorm2d(nn.Module):
    # this is the MUNIT/FUNIT papers implementation
    def __init__(self, num_channels):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.class_mean = None
        self.class_std = None
        self.num_channels = num_channels
        self.param_free_norm = nn.InstanceNorm2d(num_channels, affine=False)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N = size[:1]
        feat_var = feat.view(N, self.num_channels, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(-1, self.num_channels, 1, 1)
        feat_mean = feat.view(N, self.num_channels, -1).mean(dim=2).view(-1, self.num_channels, 1, 1)
        return feat_mean, feat_std

    def forward(self, feat):
        # input_mean, input_std = self.calc_mean_std(feat)
        # assert input_mean.shape[1] == self.num_channels and input_std.shape[1] == self.num_channels, "expected input to have {} channels instead of {}".format(self.num_channels, input_mean.shape[1])
        # assert self.class_mean.shape[1] == self.num_channels and self.class_std.shape[1] == self.num_channels, "expected AdaIN sigma & mu to have {} channels instead of {}".format(self.num_channels, self.target_mean.shape[1])
        #
        # normalized_feat = (feat - input_mean.expand(size)) / input_std.expand(size)
        normalized_feat = self.param_free_norm(feat)
        return normalized_feat * self.class_std.view(-1, self.num_channels, 1, 1) + self.class_mean.view(-1, self.num_channels, 1, 1)

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, fc_dim, n_fc,
                 weight_norm=False, activation='relu', normalize_mlp=False):#, pixel_norm=False):
        super(MLP, self).__init__()
        if weight_norm:
            linear = EqualLinear
        else:
            linear = nn.Linear

        if activation == 'lrelu':
            actvn = nn.LeakyReLU(0.2,True)
        elif activation == 'blrelu':
            actvn = BidirectionalLeakyReLU()
        else:
            actvn = nn.ReLU(True)

        self.input_dim = input_dim
        self.model = []

        # normalize input
        if normalize_mlp:
            self.model += [PixelNorm()]

         # set the first layer
        self.model += [linear(input_dim, fc_dim),
                       actvn]
        if normalize_mlp:
            self.model += [PixelNorm()]

        # set the inner layers
        for i in range(n_fc - 2):
            self.model += [linear(fc_dim, fc_dim),
                           actvn]
            if normalize_mlp:
                self.model += [PixelNorm()]

        # set the last layer
        self.model += [linear(fc_dim, out_dim)] # no output activations

        # normalize output
        if normalize_mlp:
            self.model += [PixelNorm()]

        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        out = self.model(input)
        return out

class StyledConvBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc=3, latent_dim=256, padding='reflect', upsample=False, downsample=False,
                 norm='adain', actvn='lrelu', truncate_std=False, use_pixel_norm=False, residual=False, conv_weight_norm=True,
                 normalize_affine_output=False, modulated_conv=False):
        super(StyledConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            if modulated_conv:
                conv2d = ModulatedConv2d
            else:
                conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if modulated_conv and (actvn == 'relu' or actvn == 'lrelu'):
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.norm = norm
        self.use_pixel_norm = use_pixel_norm
        self.residual = residual and (fin == fout)
        self.upsample = upsample
        self.downsample = downsample
        self.modulated_conv = modulated_conv
        if norm == 'adain' and not modulated_conv:
            norm_layer = AdaIN2d(fout, latent_dim=latent_dim, conv_weight_norm=True, truncate_std=truncate_std,
                                 normalize_affine_output=normalize_affine_output)
        elif norm == 'adaspade':
            norm_layer = AdaSPADE(fout, label_nc=semantic_nc, latent_dim=latent_dim, padding_type='reflect',
                         conv_weight_norm=True, actvn=actvn)
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d(fout)
        elif norm == 'layer':
            norm_layer = LayerNorm(fout)
        else:
            norm_layer = None

        if actvn == 'blrelu':
            activation = BidirectionalLeakyReLU()
        else:
            activation = nn.LeakyReLU(0.2,True)

        if self.residual or (not modulated_conv):
            if self.upsample:
                self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        if self.downsample:
            self.downsampler = nn.AvgPool2d(2)

        if self.modulated_conv:
            self.conv0 = conv2d(fin, fout, kernel_size=3, padding_type=padding, upsample=upsample,
            latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, fout, kernel_size=3)
            if self.upsample:
                seq0 = [self.upsampler, padding_layer(1), conv0, Blur(fout)]
            else:
                seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        if norm != 'none':
            self.norm0 = norm_layer
        if use_pixel_norm:
            self.pxl_norm0 = PixelNorm()

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(fout, fout, kernel_size=3, padding_type=padding, downsample=downsample,
            latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(fout, fout, kernel_size=3)
            if self.downsample:
                seq1 = [Blur(fout), padding_layer(1), conv1, self.downsampler]
            else:
                seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        if norm != 'none':
            self.norm1 = norm_layer
        if use_pixel_norm:
            self.pxl_norm1 = PixelNorm()

        self.actvn1 = activation

        if self.residual and self.downsample:
            self.in_downsample = nn.Sequential(Blur(fin), self.downsampler)

        if self.residual and self.upsample:
            self.in_upsample = nn.Sequential(self.upsampler, Blur(fout))

    def forward(self, input, seg=None, latent=None):
        if self.modulated_conv:
            out = self.conv0(input,latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if not self.modulated_conv and (self.norm == 'adain' or self.norm == 'adspade'):
            out = self.norm0(out, seg, latent)
        elif self.norm == 'layer' or self.norm == 'instance':
            out = self.norm0(out)
        if self.use_pixel_norm:
            out = self.pxl_norm0(out)

        if self.modulated_conv:
            out = self.conv1(out,latent)
        else:
            out = self.conv1(out)

        out = self.actvn1(out) * self.actvn_gain

        if not self.modulated_conv and (self.norm == 'adain' or self.norm == 'adspade'):
            out = self.norm1(out, seg, latent)
        elif self.norm == 'layer' or self.norm == 'instance':
            out = self.norm1(out)

        if self.residual:
            if self.upsample:
                input = self.in_upsample(input)
            if self.downsample:
                input = self.in_downsample(input)

            out = input + out

        if self.use_pixel_norm:
            out = self.pxl_norm1(out)

        return out

class AdaSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc=3, latent_dim=256, padding='reflect',
                 use_spectral_norm=False, conv_weight_norm=False, actvn='relu'):
        super(AdaSPADEResnetBlock, self).__init__()
        assert use_spectral_norm and conv_weight_norm, "Only one of use_spectral_norm and conv_weight_norm can be set to true"
        # padding
        if padding == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        conv_0 = conv2d(fin, fmiddle, kernel_size=3)
        conv_1 = conv2d(fmiddle, fout, kernel_size=3)
        if self.learned_shortcut:
            self.conv_s = conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if use_spectral_norm:
            conv_0 = spectral_norm(conv_0)
            conv_1 = spectral_norm(conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.conv_0 = nn.Sequential(padding_layer(1), conv_0)
        self.conv_1 = nn.Sequential(padding_layer(1), conv_1)

        # define normalization layers
        self.norm_0 = AdaSPADE(fin, semantic_nc, latent_dim, conv_weight_norm=conv_weight_norm, actvn=actvn)
        self.norm_1 = AdaSPADE(fmiddle, semantic_nc, latent_dim, conv_weight_norm=conv_weight_norm, actvn=actvn)
        if self.learned_shortcut:
            self.norm_s = AdaSPADE(fin, semantic_nc, latent_dim, conv_weight_norm=conv_weight_norm, actvn=actvn)

        if actvn == 'lrelu':
            self.actvn = nn.LeakyReLU(0.2,True)
        elif actvn == 'blrelu':
            self.actvn = BidirectionalLeakyReLU()
        else:
            self.actvn = nn.ReLU(True)

    # note the resnet block with AdaSPADE also takes in |seg, latent|,
    # the semantic segmentation map as input
    def forward(self, x, seg, latent=None):
        x_s = self.shortcut(x, seg, latent)

        # dx = self.conv_0(self.actvn(self.norm_0(x, seg, latent)))
        # dx = self.conv_1(self.actvn(self.norm_1(dx, seg, latent)))

        dx = self.actvn(self.norm_0(self.conv_0(x), seg, latent))
        dx = self.actvn(self.norm_1(self.conv_1(dx), seg, latent))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, latent):
        if self.learned_shortcut:
            # x_s = self.conv_s(self.norm_s(x, seg, latent))
            x_s = self.norm_s(self.conv_s(x), seg, latent)
        else:
            x_s = x
        return x_s

    # def actvn(self, x):
    #     return F.leaky_relu(x, 2e-1)

class AdaIN_MLP(nn.Module):
    def __init__(self, input_dim, adain_dim, n_adain_blks, fc_dim, n_fc,
                 weight_norm=False, activation='relu'):#, pixel_norm=False):
        super(AdaIN_MLP, self).__init__()
        if weight_norm:
            linear = EqualLinear
        else:
            linear = nn.Linear

        if activation == 'lrelu':
            actvn = nn.LeakyReLU(0.2,True)
        elif activation == 'blrelu':
            actvn = BidirectionalLeakyReLU()
        else:
            actvn = nn.ReLU(True)

        self.input_dim = input_dim
        self.n_adain = n_adain_blks * 2 # 2 adain normalization layers per resnet block

        # if pixel_norm:
        #     self.model = [PixelNorm()]
        # else:
        self.model = []

        # self.model += [Linear(input_dim, fc_dim),
        #                actvn]
        # for i in range(n_fc - 2):
        #     self.model += [Linear(fc_dim, fc_dim),
        #                    actvn]
        # self.model += [Linear(fc_dim, adain_dim * self.n_adain)] # no output activations
        self.model += [linear(input_dim, fc_dim)]
        for i in range(n_fc - 1):
            self.model += [actvn, linear(fc_dim, fc_dim)]
        self.model += [linear(fc_dim, adain_dim * self.n_adain)] # no output activations
        adain_halfdim = adain_dim // 2
        if weight_norm:
            for i in range(self.n_adain):
                self.model[-1].linear.bias.data[2*i*adain_halfdim:(2*i+1)*adain_halfdim] = 0
                self.model[-1].linear.bias.data[(2*i+1)*adain_halfdim:(2*i+2)*adain_halfdim] = 1
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        b = input.shape[0]
        return self.model(input).view(b,self.n_adain,-1)

class AdaINContentEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=7,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_weight_norm=False, actvn='relu'):
        assert(n_blocks >= 0)
        super(AdaINContentEncoder, self).__init__()

        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        elif actvn == 'blrelu':
            activation = BidirectionalLeakyReLU()
        else:
            activation = nn.ReLU(True)

        encoder = [padding_layer(3), conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [padding_layer(1),
                        conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                        norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_weight_norm=conv_weight_norm)]

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        return self.encoder(input)

class AdaINStyleEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=4, style_dim=100, padding_type='reflect',
                 mask_featues=False, vae_style_encoder=False, conv_weight_norm=False, actvn='relu'):
        super(AdaINStyleEncoder, self).__init__()

        self.mask_features = mask_featues
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        encoder = [padding_layer(3), conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [padding_layer(1),
                        conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                        activation]

        if vae_style_encoder:
            encoder += [conv2d(ngf * mult * 2, 2, kernel_size=1, stride=1, padding=0)]
        else:
            encoder += [conv2d(ngf * mult * 2, style_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*encoder)

        self.scale_factor = 1 / n_downsampling
        self.downsampler = F.interpolate

    def forward(self, input, mask=None):
        latent = self.encoder(input)
        if self.mask_features:
            assert mask != None, "mask cannot be None in that when mask_features is True"
            downsampled_mask = self.downsampler(mask, scale_factor=self.scale_factor)
            correction = (downsampled_mask.numel()).float() / ((downsampled_mask > 0.5).sum())
        else:
            downsampled_mask, correction = 1.0, 1.0

        mean_features = (latent * downsampled_mask).mean(dim=3).mean(dim=2) * correction
        return mean_features


class AdaINDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, style_dim=100, n_downsampling=3,
                 adain_blocks=2, norm_layer='layer', padding_type='reflect',
                 out_type='rgb', adaptive_norm=AdaptiveInstanceNorm2d, conv_weight_norm=False,
                 actvn='relu', use_resblk_pixel_norm=False):
        super(AdaINDecoder, self).__init__()
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        if actvn == 'blrelu':
            activation = BidirectionalLeakyReLU()
        else:
            activation = nn.ReLU(True)

        if out_type == 'segmentation':
            out_activation = nn.Softmax2d()
        else:
            out_activation = nn.Tanh()

        if norm_layer == 'layer':
            norm = LayerNorm
        else:
            norm = PixelNorm

        decoder = []
        # AdaIN resnet blocks
        mult = 2**n_downsampling
        for i in range(adain_blocks):
            decoder += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                    norm_layer=adaptive_norm, conv_weight_norm=conv_weight_norm,
                                    use_pixel_norm=use_resblk_pixel_norm)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # decoder += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            #             norm_layer(int(ngf * mult / 2)), activation]

            decoder += [nn.Upsample(scale_factor=2),
                        padding_layer(2),
                        conv2d(ngf * mult, ngf * mult // 2, kernel_size=5, stride=1, padding=0),
                        norm(int(ngf * mult / 2)), activation]

        decoder += [padding_layer(3), conv2d(ngf, output_nc, kernel_size=7, padding=0), out_activation]
        self.decoder = nn.Sequential(*decoder)

        # self.mlp = MLP(style_dim, latent_dim, 256, 8, weight_norm=conv_weight_norm, activation=actvn)#, pixel_norm=norm_layer=='pixel')
        self.mlp = AdaIN_MLP(style_dim, ngf * (2**n_downsampling) * 2, adain_blocks, 256, 3,
                             weight_norm=conv_weight_norm, activation=actvn)#, pixel_norm=norm_layer=='pixel')

    def set_adain_params(self, adain_params):
        # assign the adain_params to the AdaIN layers in model
        counter = 0
        for m in self.decoder.modules():
            classname = m.__class__.__name__
            if classname == "AdaptiveInstanceNorm2d":
                m.class_mean = adain_params[:, counter, :m.num_channels]
                m.class_std = adain_params[:, counter, m.num_channels:2*m.num_channels]
                # m.target_mean = mean.contiguous()
                # m.target_std = std.contiguous()
                counter += 1

    def forward(self, input_features, source_style, target_style):
        # mlp_code = torch.cat((source_style, target_style - source_style), 1)
        mlp_code = target_style
        adain_params = self.mlp(mlp_code)
        self.set_adain_params(adain_params)
        out = self.decoder(input_features)

        return out

class AdaSPADEDecoder(nn.Module):
    def __init__(self, output_nc, semantic_nc=3, ngf=64, style_dim=100, latent_dim=256, n_downsampling=3,
                 padding_type='reflect', use_spectral_norm=True, conv_weight_norm=False, norm_layer='layer',
                 actvn='relu'):
        super(AdaSPADEDecoder, self).__init__()
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        if norm_layer == 'layer':
            norm = LayerNorm
        else:
            norm = PixelNorm

        if actvn == 'lrelu':
            activation = nn.LeakyReLU(0.2, True)
        if actvn == 'blrelu':
            activation = BidirectionalLeakyReLU()
        else:
            activation = nn.ReLU(True)

        if conv_weight_norm:
            use_spectral_norm = False

        mult = 2**n_downsampling
        self.AdaSPADE_Resblock_0 = AdaSPADEResnetBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim, actvn=actvn,
                                              padding=padding_type, use_spectral_norm=use_spectral_norm, conv_weight_norm=conv_weight_norm)
        self.AdaSPADE_Resblock_1 = AdaSPADEResnetBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim, actvn=actvn,
                                              padding=padding_type, use_spectral_norm=use_spectral_norm, conv_weight_norm=conv_weight_norm)
        self.AdaSPADE_Resblock_2 = AdaSPADEResnetBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim, actvn=actvn,
                                              padding=padding_type, use_spectral_norm=use_spectral_norm, conv_weight_norm=conv_weight_norm)
        self.AdaSPADE_Resblock_3 = AdaSPADEResnetBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim, actvn=actvn,
                                              padding=padding_type, use_spectral_norm=use_spectral_norm, conv_weight_norm=conv_weight_norm)

        upsample = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            upsample += [nn.Upsample(scale_factor=2),
                        padding_layer(2),
                        conv2d(ngf * mult, ngf * mult // 2, kernel_size=5, stride=1, padding=0),
                        norm_layer(int(ngf * mult / 2)), activation]

        upsample += [padding_layer(3), conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.upsample = nn.Sequential(*upsample)
        # self.AdaSPADE_Resblock_up0 = AdaSPADEResnetBlock(ngf * mult, ngf * mult // 2, semantic_nc=semantic_nc, latent_dim=latent_dim, actvn=actvn,
        #                                         padding=padding_type, use_spectral_norm=use_spectral_norm, conv_weight_norm=conv_weight_norm)
        # self.AdaSPADE_Resblock_up1 = AdaSPADEResnetBlock(ngf * mult // 2, ngf * mult // 4, semantic_nc=semantic_nc, latent_dim=latent_dim, actvn=actvn,
        #                                         padding=padding_type, use_spectral_norm=use_spectral_norm, conv_weight_norm=conv_weight_norm)
        # self.conv_img = nn.Sequential(nn.LeakyReLU(0.2, True), padding_layer(1), Conv2d(ngf * mult // 4, 3, 3), nn.Tanh())
        #
        # self.upsample = nn.Upsample(scale_factor=2)

        # self.mlp = MLP(style_dim, latent_dim, 256, 8, weight_norm=conv_weight_norm,
        #                activation=actvn)#,pixel_norm=norm_layer=='pixel')
        self.mlp = AdaIN_MLP(style_dim, ngf * (2**n_downsampling) * 2, 4, 256, 3,
                             weight_norm=conv_weight_norm, activation=actvn)#, pixel_norm=norm_layer=='pixel')

    def set_adaspade_class_params(self, adain_params):
        # assign the adain_params to the AdaIN layers in model
        counter = 0
        for m in self.modules():
            classname = m.__class__.__name__
            if classname == "AdaSPADE":
                m.class_mean = adain_params[:, counter, :m.num_channels]
                m.class_std = adain_params[:, counter, m.num_channels:2*m.num_channels]
                counter += 1

    def forward(self, input_features, input_seg, target_style):
        adaspade_class_params = self.mlp(target_style)
        self.set_adaspade_class_params(adaspade_class_params)
        out = self.AdaSPADE_Resblock_0(input_features, input_seg)
        out = self.AdaSPADE_Resblock_1(out, input_seg)
        out = self.AdaSPADE_Resblock_2(out, input_seg)
        out = self.AdaSPADE_Resblock_3(out, input_seg)
        out = self.upsample(out)

        # latent = self.mlp(target_style)
        # out = self.AdaSPADE_Resblock_0(input_features, input_seg, latent)
        # out = self.AdaSPADE_Resblock_1(out, input_seg, latent)
        # out = self.AdaSPADE_Resblock_2(out, input_seg, latent)
        # out = self.AdaSPADE_Resblock_3(out, input_seg, latent)
        # out = self.upsample(out)
        # out = self.AdaSPADE_Resblock_up0(out, input_seg, latent)
        # out = self.upsample(out)
        # out = self.AdaSPADE_Resblock_up1(out, input_seg, latent)
        # out = self.conv_img(out)

        return out

class StyleGANDecoder(nn.Module):
    def __init__(self, output_nc, semantic_nc=3, ngf=64, style_dim=100, latent_dim=256, n_downsampling=3,
                 padding_type='reflect', norm='adain', upsample_norm='adain', actvn='lrelu', use_tanh=False, truncate_std=False,
                 adaptive_blocks=4, use_pixel_norm=False, residual_bottleneck=False, last_upconv_out_layers=-1,
                 conv_img_kernel_size=1, conv_weight_norm=True, normalize_mlp=False, modulated_conv=False, use_flow=False):
        super(StyleGANDecoder, self).__init__()
        self.adaptive_blocks = adaptive_blocks
        self.modulated_conv = modulated_conv
        self.use_flow = use_flow
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        mult = 2**n_downsampling
        if last_upconv_out_layers == -1:
            last_upconv_out_layers = ngf * mult // 4
        self.StyledConvBlock_0 = StyledConvBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                 padding=padding_type, norm=norm, actvn=actvn, truncate_std=truncate_std,
                                                 use_pixel_norm=use_pixel_norm, residual=residual_bottleneck,
                                                 conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)
        if self.adaptive_blocks > 1:
            self.StyledConvBlock_1 = StyledConvBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                     padding=padding_type, norm=norm, actvn=actvn, truncate_std=truncate_std,
                                                     use_pixel_norm=use_pixel_norm, residual=residual_bottleneck,
                                                     conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                     modulated_conv=modulated_conv)
        if self.adaptive_blocks > 2:
            self.StyledConvBlock_2 = StyledConvBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                     padding=padding_type, norm=norm, actvn=actvn, truncate_std=truncate_std,
                                                     use_pixel_norm=use_pixel_norm, residual=residual_bottleneck,
                                                     conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                     modulated_conv=modulated_conv)
        if self.adaptive_blocks > 3:
            self.StyledConvBlock_3 = StyledConvBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                     padding=padding_type, norm=norm, actvn=actvn, truncate_std=truncate_std,
                                                     use_pixel_norm=use_pixel_norm, residual=residual_bottleneck,
                                                     conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                     modulated_conv=modulated_conv)

        self.StyledConvBlock_up0 = StyledConvBlock(ngf * mult, ngf * mult // 2, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                   padding=padding_type, norm=upsample_norm, upsample=True, actvn=actvn, truncate_std=truncate_std,
                                                   use_pixel_norm=use_pixel_norm, conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)
        self.StyledConvBlock_up1 = StyledConvBlock(ngf * mult // 2, last_upconv_out_layers, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                   padding=padding_type, norm=upsample_norm, upsample=True, actvn=actvn, truncate_std=truncate_std,
                                                   use_pixel_norm=use_pixel_norm, conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)
        padding_sz = conv_img_kernel_size // 2
        self.use_tanh = use_tanh
        if self.modulated_conv:
            conv = ModulatedConv2d
        else:
            conv = EqualConv2d

        if use_tanh:
            # if self.modulated_conv:
            #     #demodulation is removed because kernel_size is 1
            #     self.conv_img = conv(last_upconv_out_layers, output_nc, conv_img_kernel_size, normalize_mlp=normalize_mlp)
            #     self.tanh = nn.Tanh()
            # else:
            self.conv_img = nn.Sequential(EqualConv2d(last_upconv_out_layers, output_nc, conv_img_kernel_size), nn.Tanh())
            # self.conv_img = nn.Sequential(padding_layer(padding_sz), conv2d(last_upconv_out_layers, output_nc, conv_img_kernel_size), nn.Tanh())
        else:
            self.conv_img = EqualConv2d(last_upconv_out_layers, output_nc, conv_img_kernel_size)
            # self.conv_img = nn.Sequential(padding_layer(padding_sz), conv2d(last_upconv_out_layers, output_nc, conv_img_kernel_size))

        if self.use_flow:
            self.StyledConvFlowBlock_0 = StyledConvBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                        padding=padding_type, norm=norm, actvn=actvn, truncate_std=truncate_std,
                                                        use_pixel_norm=use_pixel_norm, residual=residual_bottleneck,
                                                        conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                        modulated_conv=modulated_conv)
            self.StyledConvFlowBlock_1 = StyledConvBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                        padding=padding_type, norm=norm, actvn=actvn, truncate_std=truncate_std,
                                                        use_pixel_norm=use_pixel_norm, residual=residual_bottleneck,
                                                        conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                        modulated_conv=modulated_conv)
            self.StyledConvFlowBlock_2 = StyledConvBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                        padding=padding_type, norm=norm, actvn=actvn, truncate_std=truncate_std,
                                                        use_pixel_norm=use_pixel_norm, residual=residual_bottleneck,
                                                        conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                        modulated_conv=modulated_conv)
            self.StyledConvFlowBlock_3 = StyledConvBlock(ngf * mult, ngf * mult, semantic_nc=semantic_nc, latent_dim=latent_dim,
                                                        padding=padding_type, norm=norm, actvn=actvn, truncate_std=truncate_std,
                                                        use_pixel_norm=use_pixel_norm, residual=residual_bottleneck,
                                                        conv_weight_norm=conv_weight_norm, normalize_affine_output=normalize_mlp,
                                                        modulated_conv=modulated_conv)
            self.conv_flow = nn.Sequential(EqualConv2d(ngf * mult, 2, conv_img_kernel_size), nn.Tanh())
            self.sampler = nn.functional.grid_sample
            self.upsampler = nn.Upsample(scale_factor=4, mode='bilinear')

        self.mlp = MLP(style_dim, latent_dim, 256, 8, weight_norm=True, activation=actvn, normalize_mlp=normalize_mlp)

    def forward(self, input_features, input_seg=None, target_style=None, traverse=False, deploy=False, interp_step=0.5, xy=None, flow_seg=None):
        if target_style is not None:
            if traverse:
                alphas = torch.arange(1,0,step=-interp_step).view(-1,1).cuda()
                interps = len(alphas)
                orig_class_num = target_style.shape[0]
                output_classes = interps * (orig_class_num - 1) + 1
                temp_latent = self.mlp(target_style)
                latent = temp_latent.new_zeros((output_classes, temp_latent.shape[1]))
            else:
                latent = self.mlp(target_style)
        else:
            latent = None

        if traverse:
            input_features = input_features.repeat(output_classes,1,1,1)
            for i in range(orig_class_num-1):
                latent[interps*i:interps*(i+1), :] = alphas * temp_latent[i,:] + (1 - alphas) * temp_latent[i+1,:]
            latent[-1,:] = temp_latent[-1,:]
        elif deploy:
            output_classes = target_style.shape[0]
            input_features = input_features.repeat(output_classes,1,1,1)

        if self.use_flow:
            flow = self.StyledConvFlowBlock_0(input_features, input_seg, latent)
            flow = self.StyledConvFlowBlock_1(flow, input_seg, latent)
            flow = self.StyledConvFlowBlock_2(flow, input_seg, latent)
            flow = self.StyledConvFlowBlock_3(flow, input_seg, latent)
            flow = self.conv_flow(flow)
            upsampled_flow = self.upsampler(flow)
            flow = flow.permute(0, 2, 3, 1)
            upsampled_flow = upsampled_flow.permute(0, 2, 3, 1)

            decoder_input_features = self.sampler(input_features, flow, padding_mode='border')
            if xy is not None:
                warped_xy = self.sampler(xy, upsampled_flow, padding_mode='border')
            else:
                warped_xy = None
            if flow_seg is not None:
                warped_seg = self.sampler(flow_seg, upsampled_flow, mode='nearest', padding_mode='border')
            else:
                warped_seg = None

            flow = flow.permute(0,3,1,2)
        else:
            decoder_input_features = input_features
            flow = None
            warped_xy = None
            warped_seg = None

        out = self.StyledConvBlock_0(decoder_input_features, input_seg, latent)
        if self.adaptive_blocks > 1:
            out = self.StyledConvBlock_1(out, input_seg, latent)
        if self.adaptive_blocks > 2:
            out = self.StyledConvBlock_2(out, input_seg, latent)
        if self.adaptive_blocks > 3:
            out = self.StyledConvBlock_3(out, input_seg, latent)
        out = self.StyledConvBlock_up0(out, input_seg, latent)
        out = self.StyledConvBlock_up1(out, input_seg, latent)
        # if self.modulated_conv and self.use_tanh:
        #     out = self.tanh(self.conv_img(out,latent))
        # elif self.modulated_conv:
        #     out = self.conv_img(out,latent)
        # else:
        out = self.conv_img(out)

        return out, flow, warped_xy, warped_seg

class AdaINGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, style_dim=100, n_downsampling=3,
                 n_blocks=4, adaptive_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect',
                 vae_style_encoder=False, out_type='rgb', adaptive_norm='adain', upsample_norm='adain',
                 label_nc=3, conv_weight_norm=False, decoder_norm='layer', activation='relu',
                 use_style_decoder=False, use_tanh=False, truncate_std=False, use_resblk_pixel_norm=False,
                 residual_bottleneck=False, last_upconv_out_layers=-1, conv_img_kernel_size=1,
                 normalize_mlp=False, modulated_conv=False, use_flow=False):
        super(AdaINGenerator, self).__init__()
        self.adaptive_norm = adaptive_norm
        self.id_encoder = AdaINContentEncoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer,
                                              padding_type, conv_weight_norm=conv_weight_norm,
                                              actvn='relu') # replacing relu with leaky relu here causes nans and the entire training to collapse immediately
        self.age_encoder = AdaINStyleEncoder(input_nc, ngf=ngf, n_downsampling=4, style_dim=style_dim,
                                             padding_type=padding_type, vae_style_encoder=vae_style_encoder,
                                             conv_weight_norm=conv_weight_norm, actvn='lrelu')
        if use_style_decoder:
            use_pixel_norm = decoder_norm == 'pixel'
            self.decoder = StyleGANDecoder(output_nc, semantic_nc=label_nc, ngf=ngf, style_dim=style_dim,
                                           n_downsampling=n_downsampling, norm=adaptive_norm, upsample_norm=upsample_norm, use_tanh=use_tanh,
                                           actvn=activation, truncate_std=truncate_std, adaptive_blocks=adaptive_blocks,
                                           use_pixel_norm=use_pixel_norm, residual_bottleneck=residual_bottleneck,
                                           last_upconv_out_layers=last_upconv_out_layers, conv_img_kernel_size=conv_img_kernel_size,
                                           conv_weight_norm=conv_weight_norm, normalize_mlp=normalize_mlp, modulated_conv=modulated_conv,
                                           use_flow=use_flow)
        elif adaptive_norm == 'adain':
            use_resblk_pixel_norm = use_resblk_pixel_norm
            self.decoder = AdaINDecoder(output_nc, ngf=ngf, style_dim=style_dim, n_downsampling=n_downsampling,
                                        adain_blocks=adaptive_blocks, norm_layer=decoder_norm,
                                        padding_type=padding_type, out_type=out_type,
                                        conv_weight_norm=conv_weight_norm, actvn=activation,
                                        use_resblk_pixel_norm=use_resblk_pixel_norm)
        elif adaptive_norm == 'adaspade':
            self.decoder = AdaSPADEDecoder(output_nc, semantic_nc=label_nc, ngf=ngf, style_dim=style_dim,
                                           n_downsampling=n_downsampling, padding_type=padding_type,
                                           conv_weight_norm=conv_weight_norm, norm_layer=decoder_norm,
                                           actvn=activation)
        else:
            raise NotImplementedError('[%s] decoder is not found' % adaptive_norm)

    def encode(self, input, mask=None):
        if torch.is_tensor(input):
            id_features = self.id_encoder(input)
            age_features = self.age_encoder(input, mask)
            return id_features, age_features
        else:
            return None, None

    def decode(self, id_features, segmentation_map, target_age_features, traverse=False, deploy=False, interp_step=0.5, xy=None, flow_seg=None):
        if torch.is_tensor(id_features):
            if self.adaptive_norm == 'adain':
                return self.decoder(id_features, None, target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step, xy=xy, flow_seg=flow_seg)
            else:
                return self.decoder(id_features, segmentation_map, target_age_features, traverse=traverse)
        else:
            return None, None, None, None

    # def forward(self, input, segmentation_map, target_age_features):
    #     id_features = self.id_encoder(input)
    #     age_features = self.age_encoder(input)
    #     out = self.decode(id_features, segmentation_map, target_age_features)
    #     return out, id_features, age_features

    #parallel forward
    def forward(self, input, cyc_age_code, target_age_code, source_age_code=None, disc_pass=False, xy=None, seg=None):
        orig_id_features = self.id_encoder(input)
        orig_age_features = self.age_encoder(input)
        if disc_pass:
            rec_out = None
            rec_xy = None
        else:
            rec_out, _, rec_xy, _ = self.decode(orig_id_features, None, source_age_code, xy=xy)

        gen_out, gen_flow, gen_xy, gen_seg = self.decode(orig_id_features, None, target_age_code, xy=xy, flow_seg=seg)
        if disc_pass:
            fake_id_features = None
            fake_age_features = None
            cyc_out = None
            cyc_xy = None
        else:
            fake_id_features = self.id_encoder(gen_out)
            fake_age_features = self.age_encoder(gen_out)
            cyc_out, _, cyc_xy, _ = self.decode(fake_id_features, None, cyc_age_code, xy=gen_xy)
        return rec_out, gen_out, cyc_out, orig_id_features, orig_age_features, fake_id_features, fake_age_features, \
               gen_flow, gen_seg, rec_xy, cyc_xy


    def infer(self, input, segmentation_map, target_age_features, within_domain_idx=-1, traverse=False, deploy=False, interp_step=0.5):
        id_features = self.id_encoder(input)
        if within_domain_idx > -1:
            age_features = self.age_encoder(input)
            target_age_features[within_domain_idx] = age_features[within_domain_idx]

        out, _, _, _ = self.decode(id_features, segmentation_map, target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step)
        return out

class CondResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, is_flow=False, numClasses=2,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', cond_length=0):
        assert(n_blocks >= 0)
        # assert(round(n_blocks / 2) == n_blocks / 2), "n_blocks must be an even number"
        super(CondResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.is_flow = is_flow
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if self.is_flow:
            output_nc = 2
            self.sampler = nn.functional.grid_sample

        encoder = [padding_layer(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                   norm_layer(ngf),
                   nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(int(n_blocks)):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder += [nn.ConvTranspose2d(ngf * mult + cond_length, int(ngf * mult / 2),
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        for i in range(n_downsampling - 1):
            mult = 2**(n_downsampling - 1 - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]

        decoder += [padding_layer(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]

        self.decoder = nn.Sequential(*decoder)

    def forward(self, input, cond, parsing=None, mask=None, facial_hair_mask=None,
                clothing_items_mask=None, neck_mask=None, landmarks=None, xy=None,
                use_rgbxy_inputs=False, use_xy_inputs=False):

        if self.is_flow:
            if parsing is None:
                enc_in = input
            elif landmarks is None:
                enc_in = parsing
            else:
                enc_in = torch.cat((parsing,landmarks),1)

            if use_rgbxy_inputs:
                if parsing is None:
                    enc_in = torch.cat((enc_in, xy),1)
                else:
                    enc_in = torch.cat((enc_in, input, xy),1)
            elif use_xy_inputs:
                enc_in = torch.cat((enc_in, xy),1)

            enc_out = self.encoder(enc_in)
            dec_in = torch.cat((enc_out, cond), 1)
            flow = self.decoder(dec_in)
            flow = flow.permute(0, 2, 3, 1)
            if parsing is not None:
                parsing_out = self.sampler(parsing, flow, padding_mode='zeros')
            else:
                parsing_out = None

            if landmarks is not None:
                landmarks_out = self.sampler(landmarks, flow, padding_mode='zeros')
            else:
                landmarks_out = None

            if xy is not None:
                warped_xy = self.sampler(xy, flow, padding_mode='border')
            else:
                warped_xy = None

            if mask is not None:
                warped_mask = self.sampler(mask, flow, padding_mode='zeros')
            else:
                warped_mask = None

            if facial_hair_mask is not None:
                warped_facial_hair_mask = self.sampler(facial_hair_mask, flow, padding_mode='zeros')
            else:
                warped_facial_hair_mask = None

            if clothing_items_mask is not None:
                warped_clothing_items_mask = self.sampler(clothing_items_mask, flow, padding_mode='zeros')
            else:
                warped_clothing_items_mask = None

            if neck_mask is not None:
                warped_neck_mask = self.sampler(neck_mask, flow, padding_mode='zeros')
            else:
                warped_neck_mask = None

            out = self.sampler(input, flow, padding_mode='zeros')
            if mask is not None:
                if out.size(1) == 3:
                    masked_out = out * warped_mask
                else:
                    masked_out = out * warped_mask[:,0:1,:,:]
            else:
                masked_out = None
            # None is a placeholder for expanded parsings
            return out, masked_out, flow.permute(0, 3, 1, 2), warped_mask, parsing_out, warped_facial_hair_mask, \
                   warped_clothing_items_mask, warped_neck_mask, None, landmarks_out, warped_xy
        else:
            enc_out = self.encoder(input)
            dec_in = torch.cat((enc_out, cond), 1)
            dec_out = self.decoder(dec_in)
            return dec_out

class CondResnetFlowGenerator(nn.Module):
    def __init__(self, input_nc, ngf=64, numClasses=2,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', cond_length=0):
        assert(n_blocks >= 0)
        # assert(round(n_blocks / 2) == n_blocks / 2), "n_blocks must be an even number"
        super(CondResnetFlowGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = 2
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        encoder = [padding_layer(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                   norm_layer(ngf),
                   nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(int(n_blocks)):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder += [nn.ConvTranspose2d(ngf * mult + cond_length, int(ngf * mult / 2),
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        for i in range(n_downsampling - 1):
            mult = 2**(n_downsampling - 1 - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]

        decoder += [padding_layer(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]

        self.decoder = nn.Sequential(*decoder)
        self.sampler = nn.functional.grid_sample

    def forward(self, input, cond, mask=None, parsing=None):
        enc_out = self.encoder(input)
        dec_in = torch.cat((enc_out, cond), 1)
        dec_out = self.decoder(dec_in)
        return dec_out

        if parsing is None:
            enc_out = self.encoder(input)
            dec_in = torch.cat((enc_out, cond), 1)
            flow = self.decoder(dec_in)
            parsing_out = None
        else:
            enc_out = self.encoder(input)
            dec_in = torch.cat((enc_out, cond), 1)
            flow = self.decoder(parsing, flow_cond)

        flow = flow.permute(0, 2, 3, 1)
        if parsing is not None:
            parsing_out = self.sampler(parsing, flow, padding_mode='zeros')

        if mask is not None:
            warped_mask = self.sampler(mask, flow, padding_mode='zeros')
        else:
            warped_mask = None

        out = self.sampler(input, flow, padding_mode='zeros')
        if mask is not None:
            masked_out = out * warped_mask
        else:
            masked_out = None

        return out, masked_out, flow.permute(0, 3, 1, 2), warped_mask, parsing_out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False,
                 res_block_const=1.0, conv_weight_norm=False, use_pixel_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation,
                                                use_dropout, conv_weight_norm, use_pixel_norm)
        self.const = res_block_const

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, conv_weight_norm, use_pixel_norm):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if conv_weight_norm:
            conv2d = EqualConv2d
        else:
            conv2d = nn.Conv2d

        self.use_pixel_norm = use_pixel_norm
        if self.use_pixel_norm:
            self.pixel_norm = PixelNorm()

        conv_block += [conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.const * x + self.conv_block(x)
        return out

# Define a resnet block
class CondResnetBlock(nn.Module):
    def __init__(self, dim, num_classes, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(CondResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, num_classes, padding_type, norm_layer, activation, use_dropout)
        self.dim = dim

    def build_conv_block(self, dim, num_classes, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + num_classes, dim + num_classes, kernel_size=3, padding=p),
                       norm_layer(dim + num_classes),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + num_classes, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x[:,:self.dim,:,:] + self.conv_block(x)
        if self.use_pixel_norm:
            out = self.pixel_norm(out)

        return out

class CondEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d,
                 numClasses=2, padding_type='reflect', cond_length=0, use_avg_features=False):
        super(CondEncoder, self).__init__()
        self.output_nc = output_nc
        self.use_avg_features = use_avg_features
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        encoder = [padding_layer(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                   norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                        norm_layer(ngf * mult * 2), nn.ReLU(True)]

        self.encoder = nn.Sequential(*encoder)

        ### upsample
        decoder = [nn.ConvTranspose2d(ngf * mult * 2 + cond_length, int(ngf * mult), kernel_size=3, stride=2, padding=1, output_padding=1),
                   norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        for i in range(1, n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        decoder += [padding_layer(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input, cond, labels=None):
        enc_out = self.encoder(input)
        dec_in = torch.cat((enc_out, cond), 1)
        outputs = self.decoder(dec_in)

        # label-wise average pooling
        if self.use_avg_features:
            outputs_mean = outputs.clone()
            # inst_list = np.unique(labels.cpu().numpy().astype(int))
            for i in range(labels.size()[1]):
                for b in range(input.size()[0]):
                    # indices = (labels[b:b+1] == int(i)).nonzero() # n x 4
                    indices = (labels[b:b+1, i:i+1] == 1).nonzero() # n x 4
                    for j in range(self.output_nc):
                        output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                        mean_feat = torch.mean(output_ins).expand_as(output_ins)
                        outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
            return outputs_mean
        else:
            return outputs


##############################################################################
# Discriminator
##############################################################################

# Defines the conditional PatchGAN discriminator with the specified arguments.
class AuxOutNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, im_size=128, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d,
                 normalize=False, gpu_ids=[], numClasses=2, classify_fakes=False):
        super(AuxOutNLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.num_classes = numClasses
        if classify_fakes:
            self.num_classes += 1

        use_bias = not normalize

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),#],
                    nn.LeakyReLU(0.01, True)]

        curr_dim = ndf
        for n in range(1, n_layers):
            out_dim = min(curr_dim * 2, 1024)
            sequence += [nn.Conv2d(curr_dim, out_dim, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.01, True)]

            curr_dim = min(curr_dim * 2, 1024)

        k_size = int(im_size / (2 ** n_layers))

        self.features = nn.Sequential(*sequence)

        self.gan_head = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=padw, bias=False)
        self.class_head = nn.Conv2d(curr_dim, self.num_classes, kernel_size=k_size, bias=False)
        # class_head = [nn.Conv2d(curr_dim, self.num_classes, kernel_size=1, bias=False),
        #               nn.AdaptiveAvgPool2d(1)]
        # self.class_head = nn.Sequential(*class_head)

    def forward(self, input):
        bSize = input.size(0)
        features = self.features(input)
        gan_out = self.gan_head(features)
        class_out = self.class_head(features).view(bSize,-1,self.num_classes)

        return gan_out, class_out

class PerClassDiscriminator(nn.Module):
    def __init__(self,input_nc, ndf=64, numClasses=2, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, num_init_downsample=0, getIntermFeat=False, getFinalFeat=False, use_class_head=False,
                 selective_class_loss=False, classify_fakes=False, use_disc_cond_with_class=False, mse_class_loss=False):

        super(PerClassDiscriminator, self).__init__()
        self.num_D = num_D
        self.getFinalFeat = getFinalFeat
        self.getIntermFeat = getIntermFeat
        self.use_class_head = use_class_head and not selective_class_loss

        for i in range(numClasses):
            netD = CondMultiscaleDiscriminator(input_nc, ndf, 1, n_layers, norm_layer,
                        use_sigmoid, num_D, num_init_downsample, getIntermFeat, getFinalFeat, use_class_head,
                        False, classify_fakes, use_disc_cond_with_class, mse_class_loss)

            setattr(self, 'class'+str(i), netD)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result, result_class = model(input)
            return result[1:], result_class
        else:
            result, result_class, result_feat = model(input)
            return result, result_class, result_feat

    def forward(self, input, classes):
        result = []

        if self.getFinalFeat:
            result_feat = []
        else:
            result_feat = None

        if self.use_class_head:
            result_class = []
        else:
            result_class = None

        for i,c in enumerate(classes):
            model = getattr(self, 'class'+str(c))
            temp_res_gan, temp_res_class, temp_res_feat = self.singleD_forward(model, input[i:i+1])
            if len(result) == 0:
                for j in range(self.num_D):
                    result.append(temp_res_gan[j])
            else:
                for j in range(self.num_D):
                    result[j][0] = torch.cat((result[j][0],temp_res_gan[j][0]),0)

            if self.getFinalFeat:
                if len(result_feat) == 0:
                    for j in range(self.num_D):
                        result_feat.append(temp_res_feat[j])
                else:
                    for j in range(self.num_D):
                        result_feat[j][0] = torch.cat((result_feat[j][0],temp_res_feat[j][0]),0)

            if self.use_class_head:
                if len(result_class) == 0:
                    for j in range(self.num_D):
                        result_class.append(temp_res_class[j])
                else:
                    for j in range(self.num_D):
                        result_class[j][0] = torch.cat((result_class[j][0],temp_res_class[j][0]),0)

        return result, result_class, result_feat


class CondMultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, numClasses=2, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, num_init_downsample=0, getIntermFeat=False,
                 getFinalFeat=False, use_class_head=False, selective_class_loss=False, classify_fakes=False,
                 use_disc_cond_with_class=False, mse_class_loss=False, use_norm=True):

        super(CondMultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.getFinalFeat = getFinalFeat
        self.use_class_head = use_class_head and not selective_class_loss
        self.num_init_downsample = num_init_downsample

        for i in range(num_D):
            netD = CondNLayerDiscriminator(input_nc, ndf, n_layers, norm_layer,
                        use_sigmoid, numClasses, getIntermFeat, getFinalFeat, use_class_head, selective_class_loss,
                        classify_fakes, use_disc_cond_with_class, mse_class_loss, use_norm)

            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

            setattr(self, 'layer'+str(i)+'gan_head', netD.gan_head)
            if self.use_class_head:
                setattr(self, 'layer'+str(i)+'class_head', netD.class_head)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input, gan_head, class_head=None):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            result.append(gan_head(result[-1]))
            if self.use_class_head:
                result_class = class_head(result[-2])
            else:
                result_class = None
            return result[1:], result_class
        else:
            # return [model(input)]
            result = []
            features = model(input)
            if self.getFinalFeat:
                result_feat = features
            else:
                result_feat = None
            result.append(gan_head(features))
            if self.use_class_head:
                result_class = class_head(features)
            else:
                result_class = None
            return result, result_class, result_feat

    def forward(self, input):
        num_D = self.num_D
        result = []

        if self.getFinalFeat:
            result_feat = []
        else:
            result_feat = None

        if self.use_class_head:
            result_class = []
        else:
            result_class = None

        for i in range(self.num_init_downsample):
            input = self.downsample(input)

        input_downsampled = input
        for i in range(num_D):
            # if self.getIntermFeat:
            #     model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            # else:
            #     model = getattr(self, 'layer'+str(num_D-1-i))
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+1)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))

            gan_head = getattr(self, 'layer'+str(num_D-1-i)+'gan_head')
            if self.use_class_head:
                class_head = getattr(self, 'layer'+str(num_D-1-i)+'class_head')
            else:
                class_head = None

            res_gan, res_class, res_feat = self.singleD_forward(model, input_downsampled, gan_head, class_head)
            result.append(res_gan)

            if self.getFinalFeat:
                result_feat.append(res_feat)

            if self.use_class_head:
                result_class.append(res_class)

            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)

        return result, result_class, result_feat

# Defines the PatchGAN discriminator with the specified arguments.
class CondNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, numClasses=2, getIntermFeat=False, getFinalFeat=False, use_class_head=False,
                 selective_class_loss=False, classify_fakes=False, use_disc_cond_with_class=False, mse_class_loss=False,
                 use_norm=True, padding_type='reflect'):
        super(CondNLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.getFinalFeat = getFinalFeat
        self.use_class_head = use_class_head and not selective_class_loss
        self.n_layers = n_layers
        self.use_norm = use_norm

        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        kw = 4
        padw = int(np.floor((kw-1.0)/2))

        if (use_class_head and (not use_disc_cond_with_class)) or numClasses == 1 or selective_class_loss:
            in_channels = input_nc
        else:
            in_channels = input_nc + numClasses

        sequence = [[padding_layer(padw), nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[padding_layer(padw), nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2)]]
            if self.use_norm:
                sequence += [[norm_layer(nf)]]
            sequence += [[nn.LeakyReLU(0.2, True)]]

########## MUNIT DEBUG - uncomment when done #####################
        # for n in range(1, n_layers):
        #     nf_prev = nf
        #     nf = min(nf * 2, 512)
        #     sequence += [[
        #         nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
        #         norm_layer(nf), nn.LeakyReLU(0.2, True)
        #     ]]
        #
        # nf_prev = nf
        # nf = min(nf * 2, 512)
        # sequence += [[
        #     nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
        #     norm_layer(nf),
        #     nn.LeakyReLU(0.2, True)
        # ]]
        #
        # sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        #
        # if use_sigmoid:
        #     sequence += [[nn.Sigmoid()]]
        # if selective_class_loss:
        #     self.gan_head = nn.Conv2d(nf, numClasses, kernel_size=kw, stride=1, padding=padw)
        # else:
        #     self.gan_head = nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)
##################################################################

########## MUNIT DEBUG - comment when done #######################
        if selective_class_loss:
            self.gan_head = nn.Conv2d(nf, numClasses, kernel_size=1, stride=1)
        else:
            self.gan_head = nn.Conv2d(nf, 1, kernel_size=1, stride=1)
##################################################################

        if self.use_class_head:
            if mse_class_loss:
                num_classification_outputs = 1
            elif classify_fakes:
                num_classification_outputs = numClasses+1
            else:
                num_classification_outputs = numClasses
            self.class_head = nn.Sequential(nn.Conv2d(nf, nf * 2, kernel_size=kw, stride=2, padding=padw),
                                            norm_layer(nf * 2),
                                            nn.LeakyReLU(0.2, True),
                                            nn.Conv2d(nf * 2, num_classification_outputs, kernel_size=kw, stride=2, padding=padw),
                                            nn.AdaptiveAvgPool2d(1)) # 1 extra class for fake classification

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res_gan = [input]

            # for n in range(self.n_layers+2):
            #     model = getattr(self, 'model'+str(n))
            #     res.append(model(res[-1]))

            for n in range(self.n_layers+1):
                model = getattr(self, 'model'+str(n))
                res_gan.append(model(res_gan[-1]))
                res_gan.append(self.gan_head(res_gan[-1]))
            if self.use_class_head:
                res_class = self.class_head(res_gan[-2])
            else:
                res_class = None
            return res_gan[1:], res_class
        else:
            # return self.model(input)
            features = self.model(input)
            if getFinalFeat:
                res_feat = features
            else:
                res_feat = None

            res_gan = [self.gan_head(features)]
            if self.use_class_head:
                res_class = self.class_head(features)
            else:
                res_class = None

            return res_gan, res_class, res_feat

class StyleGANDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, numClasses=2, padding_type='reflect', actvn='lrelu', is_conditional=False):
        super(StyleGANDiscriminator, self).__init__()
        self.n_layers = n_layers
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        if actvn == 'blrelu':
            activation = BidirectionalLeakyReLU()
        else:
            activation = nn.LeakyReLU(0.2,True)

        if is_conditional:
            input_nc += numClasses

        sequence = [padding_layer(0), EqualConv2d(input_nc, ndf, kernel_size=1), activation]

        nf = ndf
        for n in range(n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [StyledConvBlock(nf_prev, nf, downsample=True, norm='none', actvn=actvn)]

        self.model = nn.Sequential(*sequence)
        if is_conditional:
            output_nc = 1
        else:
            output_nc = numClasses

        self.gan_head = nn.Sequential(padding_layer(1), EqualConv2d(nf+1, nf, kernel_size=3), activation,
                                      EqualConv2d(nf, output_nc, kernel_size=4), activation)

    def minibatch_stdev(self, input):
        out_std = torch.sqrt(input.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(input.size(0), 1, 4, 4)
        out = torch.cat((input, mean_std), 1)
        return out

    def forward(self, input):
        features = self.model(input)
        res_gan = self.gan_head(self.minibatch_stdev(features))
        return res_gan, None, None

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class Vgg16(torch.nn.Module):
    #Vgg16 as used in MUNIT (and probably FUNIT)
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.vgg16 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(30):
            self.vgg16.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu5_3 = self.vgg16(X)
        return h_relu5_3
