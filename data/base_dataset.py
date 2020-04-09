### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, normalize=True):
    transform_list = []
    use_masks = opt.use_masks
    use_parsings = opt.use_parsings
    parsing_labels_num = opt.parsing_labels_num

    if use_parsings:
        mode = Image.NEAREST
    else:
        mode = Image.BICUBIC

    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Lambda(npScale(osize, mode)))
        transform_list.append(transforms.Lambda(npRandomCrop(opt.fineSize)))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.Lambda(npRandomCrop(opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(scale_width(img, opt.loadSize)))
        transform_list.append(transforms.Lambda(npRandomCrop(opt.fineSize)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(npRandomHorizontalFlip()))

    transform_list += [transforms.ToTensor()]

    if normalize:
        num_channels = 3 + int(use_masks) + int(use_parsings) * parsing_labels_num
        mean = (0.5,)# * num_channels  # creates a length num channels tuple of 0.5's
        std = (0.5,)# * num_channels  # creates a length num channels tuple of 0.5's
        transform_list += [transforms.Normalize(mean,std)]

    return transforms.Compose(transform_list)

# def normalize():
#     return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#
# def __make_power_2(img, base, method=Image.BICUBIC):
#     ow, oh = img.size
#     h = int(round(oh / base) * base)
#     w = int(round(ow / base) * base)
#     if (h == oh) and (w == ow):
#         return img
#     return img.resize((w, h), method)

class scale_width(object):
    def __init__(self, target_width, method):
        self.target_width = target_width
        self.method = method

    def __call__(self, img):
        ow, oh = img.size
        if (ow == self.target_width):
            return img
        w = self.target_width
        h = int(self.target_width * oh / ow)
        return img.resize((w, h), self.method)

# def __crop(img, pos, size):
#     ow, oh = img.size
#     x1, y1 = pos
#     tw = th = size
#     if (ow > tw or oh > th):
#         return img.crop((x1, y1, x1 + tw, y1 + th))
#     return img
#
# def __flip(img, flip):
#     if flip:
#         return img.transpose(Image.FLIP_LEFT_RIGHT)
#     return img

#define transformation functions for numpy nd-arrays representing images with more than 3 channels
class npScale(object):
    def __init__(self, sz, mode):
        transforms_list = [transforms.ToPILImage(),
                           transforms.Resize(sz, mode)]
        self.sz = sz
        self.sc = transforms.Compose(transforms_list)

    def __call__(self, img):
        h, w, c = img.shape
        counter = 0
        while c - counter >= 3:
            temp_img = self.sc(img[:, :, counter:counter + 3])
            temp_img = np.array(temp_img.getdata(), dtype=np.uint8).reshape(self.sz[0], self.sz[1], 3)
            if counter == 0:
                out_img = temp_img
            else:
                out_img = np.concatenate((out_img, temp_img), axis=2)
            counter += 3

        while c - counter > 0:
            temp_img = self.sc(img[:, :, counter:counter + 1])
            temp_img = np.array(temp_img.getdata(), dtype=np.uint8).reshape(self.sz[0], self.sz[1], 1)
            if counter == 0:
                out_img = temp_img
            else:
                out_img = np.concatenate((out_img, temp_img), axis=2)
            counter += 1

        return out_img

class npRandomCrop(object):
    def __init__(self, out_size):
        self.th, self.tw = out_size, out_size

    def __call__(self, img):
        w, h = img.shape[0], img.shape[1]
        if w <= self.tw and h <= self.th:
            return img

        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)
        if img.ndim == 2:
            return img[i:i+self.th, j+j+self.tw]
        else:
            return img[i:i+self.th, j:j+self.tw, :]

class npRandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
                return np.fliplr(img).copy()
        return img
