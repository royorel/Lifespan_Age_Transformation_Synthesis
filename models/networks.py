### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import functools
from torch.autograd import grad as Grad
from torch.autograd import Function
import numpy as np
from math import sqrt
from pdb import set_trace as st

###############################################################################
# Functions
###############################################################################
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
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
    if norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'pixel':
        norm_layer = PixelNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, n_downsample_global=2,
             id_enc_norm='pixel', gpu_ids=[], padding_type='reflect',
             style_dim=50, init_type='gaussian',
             conv_weight_norm=False, decoder_norm='pixel', activation='lrelu',
             adaptive_blocks=4, normalize_mlp=False, modulated_conv=False):

    id_enc_norm = get_norm_layer(norm_type=id_enc_norm)

    netG = Generator(input_nc, output_nc, ngf, n_downsampling=n_downsample_global,
                     id_enc_norm=id_enc_norm, padding_type=padding_type, style_dim=style_dim,
                     conv_weight_norm=conv_weight_norm, decoder_norm=decoder_norm,
                     actvn=activation, adaptive_blocks=adaptive_blocks,
                     normalize_mlp=normalize_mlp, modulated_conv=modulated_conv)

    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])

    netG.apply(weights_init(init_type))

    return netG

def define_D(input_nc, ndf, n_layers=6, numClasses=2, gpu_ids=[],
             init_type='gaussian'):

    netD = StyleGANDiscriminator(input_nc, ndf=ndf, n_layers=n_layers,
                                 numClasses=numClasses)

    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])

    netD.apply(weights_init('gaussian'))

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
# Losses
##############################################################################
class FeatureConsistency(nn.Module):
    def __init__(self):
        super(FeatureConsistency, self).__init__()

    def __call__(self,input,target):
        return torch.mean(torch.abs(input - target))


class R1_reg(nn.Module):
    def __init__(self, lambda_r1=10.0):
        super(R1_reg, self).__init__()
        self.lambda_r1 = lambda_r1

    def __call__(self, d_out, d_in):
        """Compute gradient penalty: (L2_norm(dy/dx))**2."""
        b = d_in.shape[0]
        dydx = torch.autograd.grad(outputs=d_out.mean(),
                                   inputs=d_in,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx_sq = dydx.pow(2)
        assert (dydx_sq.size() == d_in.size())
        r1_reg = dydx_sq.sum() / b

        return r1_reg * self.lambda_r1


class SelectiveClassesNonSatGANLoss(nn.Module):
    def __init__(self):
        super(SelectiveClassesNonSatGANLoss, self).__init__()
        self.sofplus = nn.Softplus()

    def __call__(self, input, target_classes, target_is_real, is_gen=False):
        bSize = input.shape[0]
        b_ind = torch.arange(bSize).long()
        relevant_inputs = input[b_ind, target_classes, :, :]
        if target_is_real:
            loss = self.sofplus(-relevant_inputs).mean()
        else:
            loss = self.sofplus(relevant_inputs).mean()

        return loss

##############################################################################
# Generator
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
    def __init__(self, fin, fout, latent_dim=256, padding='reflect', upsample=False, downsample=False,
                 actvn='lrelu', use_pixel_norm=False, normalize_affine_output=False, modulated_conv=False):
        super(StyledConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.use_pixel_norm = use_pixel_norm
        self.upsample = upsample
        self.downsample = downsample
        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2,True)

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

        if use_pixel_norm:
            self.pxl_norm1 = PixelNorm()

        self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input,latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain
        if self.use_pixel_norm:
            out = self.pxl_norm0(out)

        if self.modulated_conv:
            out = self.conv1(out,latent)
        else:
            out = self.conv1(out)

        out = self.actvn1(out) * self.actvn_gain
        if self.use_pixel_norm:
            out = self.pxl_norm1(out)

        return out

class IdentityEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=7,
                 norm_layer=PixelNorm, padding_type='reflect',
                 conv_weight_norm=False, actvn='relu'):
        assert(n_blocks >= 0)
        super(IdentityEncoder, self).__init__()

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

class AgeEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=4, style_dim=50, padding_type='reflect',
                 conv_weight_norm=False, actvn='lrelu'):
        super(AgeEncoder, self).__init__()

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

        encoder += [conv2d(ngf * mult * 2, style_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        features = self.encoder(input)
        latent = features.mean(dim=3).mean(dim=2)
        return latent

class StyledDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, style_dim=50, latent_dim=256, n_downsampling=2,
                 padding_type='reflect', actvn='lrelu', use_tanh=True, use_pixel_norm=False,
                 normalize_mlp=False, modulated_conv=False):
        super(StyledDecoder, self).__init__()
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        mult = 2**n_downsampling
        last_upconv_out_layers = ngf * mult // 4

        self.StyledConvBlock_0 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_1 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_2 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_3 = StyledConvBlock(ngf * mult, ngf * mult, latent_dim=latent_dim,
                                                 padding=padding_type, actvn=actvn,
                                                 use_pixel_norm=use_pixel_norm,
                                                 normalize_affine_output=normalize_mlp,
                                                 modulated_conv=modulated_conv)

        self.StyledConvBlock_up0 = StyledConvBlock(ngf * mult, ngf * mult // 2, latent_dim=latent_dim,
                                                   padding=padding_type, upsample=True, actvn=actvn,
                                                   use_pixel_norm=use_pixel_norm,
                                                   normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)
        self.StyledConvBlock_up1 = StyledConvBlock(ngf * mult // 2, last_upconv_out_layers, latent_dim=latent_dim,
                                                   padding=padding_type, upsample=True, actvn=actvn,
                                                   use_pixel_norm=use_pixel_norm,
                                                   normalize_affine_output=normalize_mlp,
                                                   modulated_conv=modulated_conv)

        self.conv_img = nn.Sequential(EqualConv2d(last_upconv_out_layers, output_nc, 1), nn.Tanh())
        self.mlp = MLP(style_dim, latent_dim, 256, 8, weight_norm=True, activation=actvn, normalize_mlp=normalize_mlp)

    def forward(self, id_features, target_age=None, traverse=False, deploy=False, interp_step=0.5):
        if target_age is not None:
            if traverse:
                alphas = torch.arange(1,0,step=-interp_step).view(-1,1).cuda()
                interps = len(alphas)
                orig_class_num = target_age.shape[0]
                output_classes = interps * (orig_class_num - 1) + 1
                temp_latent = self.mlp(target_age)
                latent = temp_latent.new_zeros((output_classes, temp_latent.shape[1]))
            else:
                latent = self.mlp(target_age)
        else:
            latent = None

        if traverse:
            id_features = id_features.repeat(output_classes,1,1,1)
            for i in range(orig_class_num-1):
                latent[interps*i:interps*(i+1), :] = alphas * temp_latent[i,:] + (1 - alphas) * temp_latent[i+1,:]
            latent[-1,:] = temp_latent[-1,:]
        elif deploy:
            output_classes = target_age.shape[0]
            id_features = id_features.repeat(output_classes,1,1,1)

        out = self.StyledConvBlock_0(id_features, latent)
        out = self.StyledConvBlock_1(out, latent)
        out = self.StyledConvBlock_2(out, latent)
        out = self.StyledConvBlock_3(out, latent)
        out = self.StyledConvBlock_up0(out, latent)
        out = self.StyledConvBlock_up1(out, latent)
        out = self.conv_img(out)

        return out

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, style_dim=50, n_downsampling=2,
                 n_blocks=4, adaptive_blocks=4, id_enc_norm=PixelNorm,
                 padding_type='reflect', conv_weight_norm=False,
                 decoder_norm='pixel', actvn='lrelu', normalize_mlp=False,
                 modulated_conv=False):
        super(Generator, self).__init__()
        self.id_encoder = IdentityEncoder(input_nc, ngf, n_downsampling, n_blocks, id_enc_norm,
                                          padding_type, conv_weight_norm=conv_weight_norm,
                                          actvn='relu') # replacing relu with leaky relu here causes nans and the entire training to collapse immediately
        self.age_encoder = AgeEncoder(input_nc, ngf=ngf, n_downsampling=4, style_dim=style_dim,
                                      padding_type=padding_type, actvn=actvn,
                                      conv_weight_norm=conv_weight_norm)

        use_pixel_norm = decoder_norm == 'pixel'
        self.decoder = StyledDecoder(output_nc, ngf=ngf, style_dim=style_dim,
                                     n_downsampling=n_downsampling, actvn=actvn,
                                     use_pixel_norm=use_pixel_norm,
                                     normalize_mlp=normalize_mlp,
                                     modulated_conv=modulated_conv)

    def encode(self, input):
        if torch.is_tensor(input):
            id_features = self.id_encoder(input)
            age_features = self.age_encoder(input)
            return id_features, age_features
        else:
            return None, None

    def decode(self, id_features, target_age_features, traverse=False, deploy=False, interp_step=0.5):
        if torch.is_tensor(id_features):
            return self.decoder(id_features, target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step)
        else:
            return None

    #parallel forward
    def forward(self, input, target_age_code, cyc_age_code, source_age_code, disc_pass=False):
        orig_id_features = self.id_encoder(input)
        orig_age_features = self.age_encoder(input)
        if disc_pass:
            rec_out = None
        else:
            rec_out = self.decode(orig_id_features, source_age_code)

        gen_out = self.decode(orig_id_features, target_age_code)
        if disc_pass:
            fake_id_features = None
            fake_age_features = None
            cyc_out = None
        else:
            fake_id_features = self.id_encoder(gen_out)
            fake_age_features = self.age_encoder(gen_out)
            cyc_out = self.decode(fake_id_features, cyc_age_code)
        return rec_out, gen_out, cyc_out, orig_id_features, orig_age_features, fake_id_features, fake_age_features


    def infer(self, input, target_age_features, traverse=False, deploy=False, interp_step=0.5):
        id_features = self.id_encoder(input)
        out = self.decode(id_features, target_age_features, traverse=traverse, deploy=deploy, interp_step=interp_step)
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True),
                 conv_weight_norm=False, use_pixel_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation,
                                                conv_weight_norm, use_pixel_norm)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, conv_weight_norm, use_pixel_norm):
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
        out = x + self.conv_block(x)
        return out

##############################################################################
# Discriminator
##############################################################################
class StyleGANDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, numClasses=2, padding_type='reflect'):
        super(StyleGANDiscriminator, self).__init__()
        self.n_layers = n_layers
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d
        else:
            padding_layer = nn.ZeroPad2d

        activation = nn.LeakyReLU(0.2,True)

        sequence = [padding_layer(0), EqualConv2d(input_nc, ndf, kernel_size=1), activation]

        nf = ndf
        for n in range(n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [StyledConvBlock(nf_prev, nf, downsample=True, actvn=activation)]

        self.model = nn.Sequential(*sequence)

        output_nc = numClasses
        self.gan_head = nn.Sequential(padding_layer(1), EqualConv2d(nf+1, nf, kernel_size=3), activation,
                                      EqualConv2d(nf, output_nc, kernel_size=4), activation)

    def minibatch_stdev(self, input):
        out_std = torch.sqrt(input.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(input.size(0), 1, input.size(2), input.size(3))
        out = torch.cat((input, mean_std), 1)
        return out

    def forward(self, input):
        features = self.model(input)
        gan_out = self.gan_head(self.minibatch_stdev(features))
        return gan_out
