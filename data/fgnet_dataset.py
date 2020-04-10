import os.path
import re
import pickle
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random
import numpy as np
import util.util as util


TEX_CLASSES_UPPER_BOUNDS = [2, 6, 9, 14, 19, 29, 39, 49, 69, 120]


class FGNET_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        opt.batchSize = 1
        self.prev_A = -1
        self.prev_B = -1
        self.class_A = -1
        self.class_B = -1

        self.name_mapping = {'0-2': 0, '3-6': 1, '7-9': 2, '10-14': 3, '15-19': 4, '20-29': 5, '30-39': 6, '40-49': 7, '50-69': 8, '70-120': 9}
        self.classNames = ['0-2', '3-6', '7-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-69', '70-120']

        self.active_classes_mapping = {}
        self.inv_active_classes_mapping = {}

        # for i, name in enumerate(self.classNames):
        for i, name in enumerate(self.opt.sort_order):
            self.active_classes_mapping[i] = self.name_mapping[name]
            self.inv_active_classes_mapping[self.name_mapping[name]] = i

        opt.numClasses = len(self.active_classes_mapping)
        opt.active_classes_mapping = self.active_classes_mapping

        # arrange directories
        self.img_paths = []
        self.parsing_paths = []

        self.img_paths, self.parsing_paths = make_dataset(opt.dataroot, self.opt)

        self.ids = []
        for curr_path in self.img_paths:
            last_slash = curr_path.rfind('/')
            age_token = curr_path.rfind('A')
            if age_token == -1:
                age_token = curr_path.rfind('a')

            curr_id = curr_path[last_slash+1:age_token]
            if curr_id not in self.ids:
                self.ids += [curr_id]

        self.transform = get_transform(opt)

    def age2class(self, age):
        if self.num_flow_classes == 2:
            cluster_boundaries = [self.ages[0] + 1]
        else:
            cluster_boundaries = [3,7,10,15,20,30,40,50,70]

        cluster = 0
        for i in range(len(cluster_boundaries)):
            if age < cluster_boundaries[i]:
                break
            else:
                cluster += 1

        age_class = self.inv_active_classes_mapping.get(cluster,-1)
        return age_class

    def mask_image(self, img, parsings):
        labels_to_mask = [0,14,15,16,18]
        for idx in labels_to_mask:
            img[parsings == idx] = 127.5

        return img

    # def rescale(self, img, is_rgb=True, method=Image.BICUBIC):
    #     iw, ih = img.size
    #     if is_rgb:
    #         oc = 3
    #     else:
    #         oc = 1
    #
    #     if iw == self.opt.fineSize and ih == self.opt.fineSize:
    #         return np.array(img.getdata(), dtype=np.uint8).reshape(ih, iw, oc)
    #
    #     if ih >= iw:
    #         ow = int(self.opt.fineSize * iw / ih)
    #         oh = self.opt.fineSize
    #     else:
    #         ow = self.opt.fineSize
    #         oh = int(self.opt.fineSize * ih / iw)
    #
    #     new_im = np.array((img.resize((ow, oh), method)).getdata(), dtype=np.uint8).reshape(oh, ow, oc)
    #     # make the image square
    #     aspect_ratio = float(new_im.shape[0])/float(new_im.shape[1])
    #
    #     if aspect_ratio > 1: #more rows than columns
    #         diff = float(new_im.shape[0] - new_im.shape[1])
    #         first_pad = np.zeros((new_im.shape[0], int(np.floor(diff/2)), new_im.shape[2]), dtype=np.uint8)
    #         second_pad = np.zeros((new_im.shape[0], int(np.ceil(diff/2)), new_im.shape[2]), dtype=np.uint8)
    #         new_im = np.concatenate((first_pad, new_im, second_pad), axis=1)
    #     elif aspect_ratio < 1: #more columns than rows
    #         diff = float(new_im.shape[1] - new_im.shape[0])
    #         first_pad = np.zeros((int(np.floor(diff/2)), new_im.shape[1], new_im.shape[2]), dtype=np.uint8)
    #         second_pad = np.zeros((int(np.ceil(diff/2)), new_im.shape[1], new_im.shape[2]), dtype=np.uint8)
    #         new_im = np.concatenate((first_pad, new_im, second_pad), axis=0)
    #
    #     return new_im

    def __getitem__(self, index):
        curr_id = self.ids[index]
        imgs_list = [val for val in self.img_paths if curr_id in val]
        parsing_list = [val for val in self.parsing_paths if curr_id in val]

        out_imgs = None
        out_facial_hair = 0
        out_clothing_items = 0
        out_neck = 0
        img_classes = []
        valid_paths = []
        valid = []

        for i in range(len(imgs_list)):
            curr_img = imgs_list[i]
            curr_parsing = parsing_list[i]

            age_token = curr_img.rfind('A')
            if age_token == -1:
                age_token = curr_img.rfind('a')
            age = int(curr_img[age_token+1:age_token+3])
            img_class = self.age2class(age)

            img_classes += [img_class]
            valid_paths += [imgs_list[i]]
            valid += [True]

            img = Image.open(curr_img).convert('RGB')
            img = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 1)

            parsing = Image.open(curr_parsing).convert('L')
            parsing = np.array(parsing.getdata(), dtype=np.uint8).reshape(parsing.size[1], parsing.size[0], 1)
            img = self.mask_image(img, parsing)

            img = self.transform(img).unsqueeze(0)

            if out_imgs is None:
                out_imgs = img
            else:
                out_imgs = torch.cat((out_imgs, img), 0)

        img_classes = torch.LongTensor(img_classes)
        valid = torch.ByteTensor(valid)

        return {'Imgs': out_imgs,
                'Paths': valid_paths,
                'Classes': img_classes,
                'Valid': valid}

    def __len__(self):
        # return len(self.img_paths)
        return len(self.ids)

    def name(self):
        return 'FGNET-Dataset'
