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

FLOW_CLASSES_UPPER_BOUNDS = [2, 6, 10, 15, 20, 120]
FLOW_CLASSES_UPPER_BOUNDS_7_CLASSES = [2, 6, 10, 15, 20, 50, 120]
TEX_CLASSES_UPPER_BOUNDS = [2, 6, 10, 15, 20, 30, 40, 50, 70, 120]

class FGNET_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        opt.batchSize = 1
        self.use_cropped = opt.use_cropped
        self.use_masks = opt.use_masks
        self.use_landmarks = opt.use_landmarks
        self.json_landmarks = self.use_landmarks and 'nvidia' in self.root
        opt.json_landmarks = self.json_landmarks
        self.embed_landmarks = opt.embed_landmarks
        self.use_parsings = opt.use_parsings
        self.hair_only = opt.hair_only
        self.no_facial_hair = opt.no_facial_hair
        self.no_clothing_items = opt.no_clothing_items
        self.no_neck_tex = opt.no_neck_tex
        self.parsing_labels_num = opt.parsing_labels_num
        if 'final_face_parsing_dataset' in self.root:
            self.parsing_labels_num = 15 # adjust labels num for final dataset
        self.use_flow_classes = opt.use_flow_classes
        self.num_flow_classes = opt.num_flow_classes
        self.is_frontal_dict = None
        self.mode = 'uniform_tex'
        self.prev_A = -1
        self.prev_B = -1
        self.class_A = -1
        self.class_B = -1
        self.flow_class_A = -1
        self.flow_class_B = -1

        self.name_mapping = {'0-2': 0, '3-6': 1, '7-10': 2, '11-15': 3, '16-20': 4, '21-30': 5, '31-40': 6, '41-50': 7, '51-70': 8, '71-120': 9}
        self.classNames = ['0-2', '3-6', '7-10', '11-15', '16-20', '21-30', '31-40', '41-50', '51-70', '71-120'] #self.classNames

        self.active_classes_mapping = {}
        self.inv_active_classes_mapping = {}
        self.inv_active_flow_classes_mapping = {}

        if self.num_flow_classes == 2:
            self.tex2flow_mapping = {0: 0, 1: 1}
            self.flow2tex_mapping = {0: [0], 1: [1]}
            self.name_mapping = {self.opt.sort_order[0]: 0, self.opt.sort_order[1]: 0}
            self.classNames = {self.opt.sort_order[0], self.opt.sort_order[1]}
            self.ages = [int(i) for i in self.opt.sort_order]
        elif self.num_flow_classes == 7:
            self.flow_upper_bounds = FLOW_CLASSES_UPPER_BOUNDS_7_CLASSES
            self.tex2flow_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 6, 9: 6}
            self.flow2tex_mapping = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5, 6, 7], 6: [8, 9]}
        else:
            self.flow_upper_bounds = FLOW_CLASSES_UPPER_BOUNDS
            self.tex2flow_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5}
            self.flow2tex_mapping = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5, 6, 7, 8, 9]}

        # for i, name in enumerate(self.classNames):
        for i, name in enumerate(self.opt.sort_order):
            self.active_classes_mapping[i] = self.name_mapping[name]
            self.inv_active_classes_mapping[self.name_mapping[name]] = i

        self.inv_active_flow_classes_mapping = self.inv_active_classes_mapping
        opt.numClasses = len(self.active_classes_mapping) #self.numClasses
        opt.numFlowClasses = len(self.inv_active_flow_classes_mapping) #self.numFlowClasses
        opt.tex2flow_mapping = self.tex2flow_mapping
        opt.active_classes_mapping = self.active_classes_mapping
        opt.inv_active_flow_classes_mapping = self.inv_active_flow_classes_mapping

        # arrange directories
        self.img_paths = []
        if self.opt.use_parsings is True:
            self.parsing_paths = []

        if self.opt.use_parsings is True:
            self.img_paths, self.parsing_paths = make_dataset(opt.dataroot, self.opt)
        else:
            self.img_paths = make_dataset(opt.dataroot, self.opt)

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
            cluster_boundaries = [3,7,11,16,21,31,41,51,71]

        cluster = 0
        for i in range(len(cluster_boundaries)):
            if age < cluster_boundaries[i]:
                break
            else:
                cluster += 1

        return cluster

    def expand_parsing_labels(self, labels, num_out_labels=15):
        h, w, c = labels.shape
        max_label = num_out_labels - 1
        reshaped_labels = labels.reshape((1, -1))
        out = np.zeros((num_out_labels, h*w), dtype=np.uint8)
        if num_out_labels == 15 and self.no_facial_hair:
            facial_hair_mask = np.zeros((3, h*w), dtype=np.uint8)
        if num_out_labels == 15 and self.no_facial_hair:
            clothing_items_mask = np.zeros((3, h*w), dtype=np.uint8)
        if num_out_labels == 15 and self.no_neck_tex:
            neck_mask = np.zeros((3, h*w), dtype=np.uint8)

        if self.hair_only:
            ind = (reshaped_labels == 10).reshape((h * w))
            if num_out_labels == 15:
                out[10, ind] = 255
            else:
                out[max_label, ind] = 255
        else:
            for i in range(max_label + 1):
                ind = (reshaped_labels == i).reshape((h * w))
                if num_out_labels == 2 and i == 0 :
                    continue
                elif num_out_labels == 2 and i == max_label:
                    out[1, ind] = 255
                elif num_out_labels == 2:
                    out[0, ind] = 255
                elif num_out_labels == 3 and i > 0 and i < max_label:
                    out[1, ind] = 255
                elif num_out_labels == 3 and i == max_label:
                    # make sure that the hair is always in the last channel
                    out[2, ind] = 255
                else:
                    if num_out_labels == 15 and (i == 11 or i == 13) and self.no_clothing_items:
                        out[i, ind] = 255
                        clothing_items_mask[:, ind] = 255
                    elif num_out_labels == 15 and i == 14 and self.no_neck_tex:
                        out[i, ind] = 255
                        neck_mask[:, ind] = 255
                    elif num_out_labels == 15 and i == 12 and self.no_facial_hair:
                        out[1, ind] = 255
                        facial_hair_mask[:, ind] = 255
                    else:
                        out[i, ind] = 255

        parsing_out = np.transpose(out.reshape((num_out_labels, h, w)), (1, 2, 0))
        if self.no_neck_tex:
            neck_out = np.transpose(neck_mask.reshape((3, h, w)), (1, 2, 0))
        else:
            neck_out = None

        if self.no_clothing_items:
            clothing_items_out = np.transpose(clothing_items_mask.reshape((3, h, w)), (1, 2, 0))
        else:
            clothing_items_out = None

        if self.no_facial_hair:
            facial_hair_out = np.transpose(facial_hair_mask.reshape((3, h, w)), (1, 2, 0))
        else:
            facial_hair_out = None

        return parsing_out, facial_hair_out, clothing_items_out, neck_out

    def rescale(self, img, is_rgb=True, method=Image.BICUBIC):
        iw, ih = img.size
        if is_rgb:
            oc = 3
        else:
            oc = 1

        if iw == self.opt.fineSize and ih == self.opt.fineSize:
            return np.array(img.getdata(), dtype=np.uint8).reshape(ih, iw, oc)

        if ih >= iw:
            ow = int(self.opt.fineSize * iw / ih)
            oh = self.opt.fineSize
        else:
            ow = self.opt.fineSize
            oh = int(self.opt.fineSize * ih / iw)

        new_im = np.array((img.resize((ow, oh), method)).getdata(), dtype=np.uint8).reshape(oh, ow, oc)
        # make the image square
        aspect_ratio = float(new_im.shape[0])/float(new_im.shape[1])

        if aspect_ratio > 1: #more rows than columns
            diff = float(new_im.shape[0] - new_im.shape[1])
            first_pad = np.zeros((new_im.shape[0], int(np.floor(diff/2)), new_im.shape[2]), dtype=np.uint8)
            second_pad = np.zeros((new_im.shape[0], int(np.ceil(diff/2)), new_im.shape[2]), dtype=np.uint8)
            new_im = np.concatenate((first_pad, new_im, second_pad), axis=1)
        elif aspect_ratio < 1: #more columns than rows
            diff = float(new_im.shape[1] - new_im.shape[0])
            first_pad = np.zeros((int(np.floor(diff/2)), new_im.shape[1], new_im.shape[2]), dtype=np.uint8)
            second_pad = np.zeros((int(np.ceil(diff/2)), new_im.shape[1], new_im.shape[2]), dtype=np.uint8)
            new_im = np.concatenate((first_pad, new_im, second_pad), axis=0)

        return new_im

    def __getitem__(self, index):
        curr_id = self.ids[index]
        imgs_list = [val for val in self.img_paths if curr_id in val]
        parsing_list = [val for val in self.parsing_paths if curr_id in val]

        out_imgs = None
        out_facial_hair = 0
        out_clothing_items = 0
        out_neck = 0
        img_classes = []
        flow_classes = []
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
            if img_class not in self.inv_active_classes_mapping.keys():
                continue
            flow_class = self.tex2flow_mapping[img_class]

            img_classes += [img_class]
            flow_classes += [flow_class]
            valid_paths += [imgs_list[i]]
            valid += [True]

            img = Image.open(curr_img).convert('RGB')
            img = self.rescale(img)
            if self.opt.use_parsings is True:
                parsing = Image.open(curr_parsing).convert('L')
                parsing = np.array(parsing.getdata(), dtype=np.uint8).reshape(parsing.size[1], parsing.size[0], 1)
                expanded_parsing, facial_hair_mask, clothing_items_mask, neck_mask = self.expand_parsing_labels(parsing, self.parsing_labels_num)
                parsing_img = util.parsingLabels2image(expanded_parsing/255)
                img = np.concatenate((img, parsing_img), axis=2)

            img = self.transform(img).unsqueeze(0)
            if self.no_facial_hair:
                facial_hair_mask = self.transform(facial_hair_mask).unsqueeze(0)
            else:
                facial_hair_mask = 0

            if self.no_clothing_items:
                clothing_items_mask = self.transform(clothing_items_mask).unsqueeze(0)
            else:
                clothing_items_mask = 0

            if self.no_neck_tex:
                neck_mask = self.transform(neck_mask).unsqueeze(0)
            else:
                neck_mask = 0

            if out_imgs is None:
                out_imgs = img
                if self.no_facial_hair:
                    out_facial_hair = facial_hair_mask
                if self.no_clothing_items:
                    out_clothing_items = clothing_items_mask
                if self.no_neck_tex:
                    out_neck = neck_mask
            else:
                out_imgs = torch.cat((out_imgs, img), 0)
                if self.no_facial_hair:
                    out_facial_hair = torch.cat((out_facial_hair, facial_hair_mask), 0)
                if self.no_clothing_items:
                    out_clothing_items = torch.cat((out_clothing_items, clothing_items_mask), 0)
                if self.no_neck_tex:
                    out_neck = torch.cat((out_neck, neck_mask), 0)

        img_classes = torch.LongTensor(img_classes)
        flow_classes = torch.LongTensor(flow_classes)
        valid = torch.ByteTensor(valid)

        return {'Imgs': out_imgs, 'facial_hair': out_facial_hair,
                'clothing_items': out_clothing_items, 'neck': out_neck,
                'Paths': valid_paths, 'Classes': img_classes, 'flow_Classes': flow_classes,
                'Valid': valid}

    def __len__(self):
        # return len(self.img_paths)
        return len(self.ids)

    def name(self):
        return 'FGNET-Dataset'
