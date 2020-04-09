from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import os
import collections
import math
from skimage import color
from pdb import set_trace as st

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    im_sz = image_tensor.size()
    ndims = image_tensor.dim()
    if ndims == 2:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
    elif ndims == 3:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    elif ndims == 4 and im_sz[0] == 1:
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    elif ndims == 4:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    else: # ndims == 5
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 1, 3, 4, 2)) + 1) / 2.0 * 255.0

    return image_numpy.astype(imtype)

def makeColorWheel():
    # taken from from the color circle idea described at
    # http://members.shaw.ca/quadibloc/other/colint.htm
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    col = 0

    # RY
    colorwheel[:RY, 0] = 255
    colorwheel[:RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def flow2im(flow_tensor, imtype=np.uint8):
    # based on Ce Liu optical flow code
    # assuming 3D, 4D or 5D input tensor
    flow_tensor = flow_tensor.cpu()
    ndims = flow_tensor.dim()
    if ndims == 3:
        ch, h, w = flow_tensor.size()
    if ndims == 4:
        bsize, ch, h, w = flow_tensor.size()
    if ndims == 5:
        cls_num, bsize, ch, h, w = flow_tensor.size()

    x = torch.linspace(-1, 1, steps=w)
    y = torch.linspace(-1, 1, steps=h)
    if ndims == 3:
        xx = x.view(1, -1).repeat(1, h, 1)
        yy = y.view(-1, 1).repeat(1, 1, w)
        flow_x, flow_y = flow_tensor[:1, :, :] - xx, flow_tensor[1:, :, :] - yy
        flow_mag = torch.sqrt(flow_x.pow(2) + flow_y.pow(2))
        flow_max = (flow_mag.max(dim=2, keepdim=True)[0]).max(dim=1, keepdim=True)[0]
        # flow_max = 0.05
        flow_mag = (flow_mag / flow_max).float().numpy()
        flow_mag = np.transpose(flow_mag, (1, 2, 0))
        flow_phase = (torch.atan2(-flow_y, -flow_x) / math.pi).float().numpy()
        flow_phase = np.transpose(flow_phase, (1, 2, 0))
        flow_rgb = np.zeros((h, w, 3), dtype=np.float32)
    if ndims == 4:
        xx = x.view(1, -1).repeat(bsize, 1, h, 1)
        yy = y.view(-1, 1).repeat(bsize, 1, 1, w)
        flow_x, flow_y = flow_tensor[:, :1, :, :] - xx, flow_tensor[:, 1:, :, :] - yy
        flow_mag = torch.sqrt(flow_x.pow(2) + flow_y.pow(2))
        flow_max = (flow_mag.max(dim=3, keepdim=True)[0]).max(dim=2, keepdim=True)[0]
        # flow_max = 0.05
        flow_mag = (flow_mag / flow_max).float().numpy()
        flow_mag = np.transpose(flow_mag, (0, 2, 3, 1))
        flow_phase = (torch.atan2(flow_y, -flow_x) / math.pi).float().numpy()
        flow_phase = np.transpose(flow_phase, (0, 2, 3, 1))
        flow_rgb = np.zeros((bsize, h, w, 3), dtype=np.float32)
    if ndims == 5:
        xx = x.view(1, -1).repeat(cls_num, bsize, 1, h, 1)
        yy = y.view(-1, 1).repeat(cls_num, bsize, 1, 1, w)
        flow_x, flow_y = flow_tensor[:, :, :1, :, :] - xx, flow_tensor[:, :, 1:, :, :] - yy
        flow_mag = torch.sqrt(flow_x.pow(2) + flow_y.pow(2))
        flow_max = (flow_mag.max(dim=4, keepdim=True)[0]).max(dim=3, keepdim=True)[0]
        # flow_max = 0.05
        flow_mag = (flow_mag / flow_max).float().numpy()
        flow_mag = np.transpose(flow_mag, (0, 1, 3, 4, 2))
        flow_phase = (torch.atan2(-flow_y, -flow_x) / math.pi).float().numpy()
        flow_phase = np.transpose(flow_phase, (0, 1, 3, 4, 2))
        flow_rgb = np.zeros((cls_num, bsize, h, w, 3), dtype=np.float32)

    colorwheel = makeColorWheel()
    ncols = colorwheel.shape[0]
    fk = ((flow_phase + 1) / 2) * ncols
    k0 = np.minimum(np.floor(fk).astype(np.int), ncols-1)
    k1 = (k0 + 1).astype(np.int)
    k1[k1 == ncols] = 0
    f = fk - k0.astype(np.float32)

    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1

        idx = flow_mag <= 1
        col[idx] = 1 - flow_mag[idx] * (1-col[idx])
        col[np.logical_not(idx)] = col[np.logical_not(idx)] * 0.75

        if ndims == 3:
            flow_rgb[:,:,i:i+1] = np.floor(255 * col)
        if ndims == 4:
            flow_rgb[:,:,:,i:i+1] = np.floor(255 * col)
        if ndims == 5:
            flow_rgb[:,:,:,:,i:i+1] = np.floor(255 * col)

    return flow_rgb.astype(imtype)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
labels_colormap = np.array([[0,   0,   0],   # background
                            [255, 255, 0],   # face skin
                            [139, 76,  57],  # left eyebrow
                            [139, 54,  38],  # right eyebrow
                            [0,   205, 0],   # eye01
                            [0,   138, 0],   # eye02
                            [154, 50,  205], # nose
                            [0,   0,   139], # lower lip
                            [255, 165, 0],   # mouth
                            [72,  118, 255], # upper lip
                            [255, 0,   0],   # hair
                            [0, 180, 128],   # clothing
                            [255, 0, 255],   # facial hair
                            [128, 0, 180],   # hair accesory
                            [201, 151, 101]])# body skin

labels_colormap19 = np.array([[  0,   0,   0], # background
                              [128,   0,   0], # face skin
                              [  0, 128,   0], # nose
                              [128, 128,   0], # eyeglasses
                              [  0,   0, 128], # left eye
                              [128,   0, 128], # right eye
                              [  0, 128, 128], # left eyebrow
                              [128, 128, 128], # right eyebrow
                              [ 64,   0, 128], # left ear
                              [192,   0,   0], # right ear
                              [ 64, 128,   0], # mouth
                              [192, 128,   0], # upper lip
                              [ 64,   0, 128], # lower lip
                              [192,   0, 128], # hair
                              [ 64, 128, 128], # hat
                              [192, 128, 128], # earring
                              [  0,  64,   0], # neckless
                              [128,  64,   0], # neck
                              [  0, 192,   0]])# clothing item

def flip_eye_labels(parsings):
    b, _, h, w = parsings.shape # only 3 color channels are assumed
    zeros = torch.zeros(b,1,h,w).cuda().byte()
    eye1_ind = torch.round(127.5*(parsings[:,1,:,:]+1)) == 205
    eye2_ind = torch.round(127.5*(parsings[:,1,:,:]+1)) == 138
    eyebrow1_ind = torch.round(127.5*(parsings[:,2,:,:]+1)) == 57
    eyebrow2_ind = torch.round(127.5*(parsings[:,2,:,:]+1)) == 38
    eye1_ind = torch.cat((zeros, eye1_ind.unsqueeze(1), zeros), 1)
    eye2_ind = torch.cat((zeros, eye2_ind.unsqueeze(1), zeros), 1)
    eyebrow1_ind1 = torch.cat((zeros, eyebrow1_ind.unsqueeze(1), zeros), 1)
    eyebrow1_ind2 = torch.cat((zeros, zeros, eyebrow1_ind.unsqueeze(1)), 1)
    eyebrow2_ind1 = torch.cat((zeros, eyebrow2_ind.unsqueeze(1), zeros), 1)
    eyebrow2_ind2 = torch.cat((zeros, zeros, eyebrow2_ind.unsqueeze(1)), 1)
    parsings[eye1_ind] = 138 / 127.5 - 1
    parsings[eye2_ind] = 205 / 127.5 - 1
    parsings[eyebrow1_ind1] = 54 / 127.5 - 1
    parsings[eyebrow1_ind2] = 38 / 127.5 - 1
    parsings[eyebrow2_ind1] = 76 / 127.5 - 1
    parsings[eyebrow2_ind2] = 57 / 127.5 - 1

    return parsings

def merge_eye_labels(parsings):
    b, _, h, w = parsings.shape # only 3 color channels are assumed
    zeros = torch.zeros(b,1,h,w).cuda().byte()
    eye2_ind = torch.round(127.5*(parsings[:,1,:,:]+1)) == 138
    eyebrow2_ind = torch.round(127.5*(parsings[:,2,:,:]+1)) == 38
    eye2_ind = torch.cat((zeros, eye2_ind.unsqueeze(1), zeros), 1)
    eyebrow2_ind1 = torch.cat((zeros, eyebrow2_ind.unsqueeze(1), zeros), 1)
    eyebrow2_ind2 = torch.cat((zeros, zeros, eyebrow2_ind.unsqueeze(1)), 1)
    parsings[eye2_ind] = 205 / 127.5 - 1
    parsings[eyebrow2_ind1] = 76 / 127.5 - 1
    parsings[eyebrow2_ind2] = 57 / 127.5 - 1

    return parsings

def restoreFacialHair(parsings, facial_hair_mask):
    b, _, h, w = parsings.shape # only 3 color channels are assumed

    # create a full magenta image (color of facial hair labal)
    facial_hair_label = torch.ones(b, 3, h, w).cuda()
    facial_hair_label[:,1,:,:] = -1

    output = parsings * (1-facial_hair_mask) + facial_hair_label * facial_hair_mask

    return output

def removeFacialHair(image, facial_hair_mask):
    b, _, h, w = image.shape # only 3 color channels are assumed

    # create a full yellow image (color of face skin labal)
    face_skin_label = torch.ones(b, 3, h, w).cuda()
    face_skin_label[:,2,:,:] = -1

    output = image * (1-facial_hair_mask) + face_skin_label * facial_hair_mask

    return output

def removeClothingItems(parsings, clothing_items_mask):
    b, _, h, w = parsings.shape # only 3 color channels are assumed

    # create a full black parsings (color of face skin labal)
    background_label = -torch.ones(b, 3, h, w).cuda()
    output = parsings * (1-clothing_items_mask) + background_label * clothing_items_mask

    return output

def removeNeck(parsings, neck_mask):
    b, _, h, w = parsings.shape # only 3 color channels are assumed

    # create a full black parsings (color of face skin labal)
    background_label = -torch.ones(b, 3, h, w).cuda()
    output = parsings * (1-neck_mask) + background_label * neck_mask

    return output

def parsingLabels2image(probs, imtype=np.uint8):
    if isinstance(probs,torch.Tensor):
        ndims = probs.dim
        if ndims == 3:
            probs = probs.permute(1,2,0).cpu().numpy()
            h, w, c = probs.shape  # c can be 2, 3, 11 or 15. 2 is skin and hair. 3 is background, skin and hair
            nt, ns = 1, 1
        elif ndims == 4 and im_sz[0] == 1:
            # probs = probs[0].permute(1,2,0).cpu().numpy()
            h, w, c = probs.shape  # c can be 2, 3, 11 or 15. 2 is skin and hair. 3 is background, skin and hair
            nt, ns = 1, 1
        elif ndims == 4:
            probs = probs.permute(0,2,3,1).cpu().numpy()
            ns, h, w, c = probs.shape  # c can be 2, 3, 11 or 15. 2 is skin and hair. 3 is background, skin and hair
            nt = 1
        else: # ndims == 5
            probs = probs.permute(0,1,3,4,2).cpu().numpy()
            nt, ns, h, w, c = probs.shape  # c can be 2, 3, 11 or 15. 2 is skin and hair. 3 is background, skin and hair
    else:
        h, w, c = probs.shape  # c can be 2, 3, 11 or 15. 2 is skin and hair. 3 is background, skin and hair
        nt, ns = 1, 1
        ndims = 3


    image = np.zeros((nt, ns, 3, h*w))
    labels = np.argmax(probs, axis=-1).reshape((nt, ns, h*w))
    if c > 15:
        colormap = labels_colormap19
    else:
        colormap = labels_colormap

    for k in range(nt):
        for j in range(ns):
            for i in range(c):
                ind = np.squeeze(labels[k,j] == i)
                if c == 2 and i == 0:
                    foreground = np.sum(probs[k,j],axis=2) >= 64
                    ind = np.squeeze(np.logical_and(ind, foreground.reshape((1, h*w))))
                    image[k,j,:, ind] = np.expand_dims(colormap[1, :], 0)
                elif i == c - 1 and c < 15: # this makes sure that the hair color is red
                    image[k,j,:, ind] = np.expand_dims(colormap[10, :], 0)
                else:
                    image[k,j,:, ind] = np.expand_dims(colormap[i, :], 0)

    image = image.reshape((nt, ns, 3, h, w))
    if ndims == 3:
        image = np.transpose(image.reshape((3, h, w)), (1, 2, 0))
    elif ndims == 4:
        image = np.transpose(image.reshape((ns, 3, h, w)), (0, 2, 3, 1))
    else:
        image = np.transpose(image.reshape((nt, ns, 3, h, w)), (0, 1, 3, 4, 2))

    return image.astype(imtype)

def probs2rgb(probs, n_labels=15):
    n, _, h, w = probs.shape
    out = torch.zeros(n,3,h,w).cuda()
    #normalize input
    max_val = probs.view(n,-1).max(dim=1)[0].view(n,1,1,1)
    min_val = probs.view(n,-1).min(dim=1)[0].view(n,1,1,1)
    probs = (probs - min_val) / (max_val - min_val)

    for i in range(n_labels):
        out[:,0:1,:,:] += probs[:,i:i+1,:,:] * ((labels_colormap[i,0] / 127.5) - 1)
        out[:,1:2,:,:] += probs[:,i:i+1,:,:] * ((labels_colormap[i,1] / 127.5) - 1)
        out[:,2:3,:,:] += probs[:,i:i+1,:,:] * ((labels_colormap[i,2] / 127.5) - 1)

    return out

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
