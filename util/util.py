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

labels_colormap = np.array([[  0,   0,   0], # background
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

def probs2rgb(probs, n_labels=19):
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
