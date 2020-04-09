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
import json
import numpy as np
import util.util as util

FLOW_CLASSES_UPPER_BOUNDS = [2, 6, 9, 14, 19, 120]
FLOW_CLASSES_UPPER_BOUNDS_7_CLASSES = [2, 6, 9, 14, 19, 49, 120]
TEX_CLASSES_UPPER_BOUNDS = [2, 6, 9, 14, 19, 29, 39, 49, 69, 120]

class MulticlassUnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        opt.json_landmarks = False
        self.embed_landmarks = False
        self.use_parsings = opt.use_parsings
        self.hair_only = False
        self.no_facial_hair = True
        self.no_clothing_items = True
        self.no_neck_tex = False
        self.merge_eyes_and_lips = False
        self.no_facial_features = opt.no_facial_features
        self.parsing_labels_num = opt.parsing_labels_num
        self.parsing_labels_num = 19
        self.use_expanded_parsings = False
        self.use_flow_classes = False
        self.num_flow_classes = opt.num_flow_classes
        self.name_mapping = {}
        self.tex2flow_mapping = {}
        self.flow2tex_mapping = {}
        self.mode = 'uniform_tex'
        self.prev_A = -1
        self.prev_B = -1
        self.class_A = -1
        self.class_B = -1
        self.flow_class_A = -1
        self.flow_class_B = -1
        self.get_samples = False
        if self.opt.isTrain:
            self.forward_pass_id_loss = opt.forward_pass_id_loss

        # find all existing classes in root
        self.tempClassNames = []
        subDirs = next(os.walk(self.root))[1]  # a quick way to get all subdirectories
        for currDir in subDirs:
            if self.opt.isTrain:
                prefix = 'train'
            else:
                prefix = 'test'
            if prefix in currDir:  # we assume that the class name starts with the prefix
                len_prefix = len(prefix)
                className = currDir[len_prefix:]
                self.tempClassNames += [className]

        # sort classes if necessary
        if self.num_flow_classes == 7:
            self.flow_upper_bounds = FLOW_CLASSES_UPPER_BOUNDS_7_CLASSES
        else:
            self.flow_upper_bounds = FLOW_CLASSES_UPPER_BOUNDS

        if self.opt.sort_classes is True:
            if len(self.opt.sort_order) > 0:
                self.classNames = []
                for i, nextClass in enumerate(self.opt.sort_order):
                    for currClass in self.tempClassNames:
                        if nextClass == currClass:
                            self.classNames += [currClass]
                            if 'single_age_dataset' in self.root:
                                tex_class, flow_class = i, i
                                self.tex2flow_mapping[i] = i
                                self.flow2tex_mapping[i] = i
                            else:
                                tex_class, flow_class = self.assign_tex_class(currClass), self.assign_flow_class(currClass)
                                self.name_mapping[currClass] = tex_class
                                self.tex2flow_mapping[tex_class] = flow_class
                                if flow_class in self.flow2tex_mapping.keys():
                                    self.flow2tex_mapping[flow_class] += [tex_class]
                                else:
                                    self.flow2tex_mapping[flow_class] = [tex_class]
            else:
                self.classNames = sorted(self.tempClassNames)
                for i, currClass in enumerate(self.classNames):
                    if 'single_age_dataset' in self.root:
                        tex_class, flow_class = i, i
                        self.tex2flow_mapping[i] = i
                        self.flow2tex_mapping[i] = i
                    else:
                        tex_class, flow_class = self.assign_tex_class(currClass), self.assign_flow_class(currClass)
                        self.name_mapping[currClass] = tex_class
                        self.tex2flow_mapping[tex_class] = flow_class
                        if flow_class in self.flow2tex_mapping.keys():
                            self.flow2tex_mapping[flow_class] += [tex_class]
                        else:
                            self.flow2tex_mapping[flow_class] = [tex_class]
        else:
            self.classNames = self.tempClassNames

        self.active_classes_mapping = {}
        self.inv_active_classes_mapping = {}

        if 'single_age_dataset' in self.root:
            for i, name in enumerate(self.classNames):
                self.active_classes_mapping[i] = i
                self.inv_active_classes_mapping[i] = i
        else:
            for i, name in enumerate(self.classNames):
                self.active_classes_mapping[i] = self.name_mapping[name]
                self.inv_active_classes_mapping[self.name_mapping[name]] = i

        if self.use_flow_classes:
            self.active_flow_classes_mapping = {}
            self.inv_active_flow_classes_mapping = {}
            if self.opt.isTrain and opt.load_pretrained_flow == '':
                counter = 0
                for i, name in enumerate(self.classNames):
                    if self.tex2flow_mapping[self.name_mapping[name]] not in self.active_flow_classes_mapping.values():
                        self.active_flow_classes_mapping[counter] = self.tex2flow_mapping[self.name_mapping[name]]
                        self.inv_active_flow_classes_mapping[self.tex2flow_mapping[self.name_mapping[name]]] = counter
                        counter += 1
            else:
                # use identity mapping on fully trained flow network
                for i, name in enumerate(self.classNames):
                    if self.tex2flow_mapping[self.name_mapping[name]] not in self.active_flow_classes_mapping.values():
                        self.active_flow_classes_mapping[self.tex2flow_mapping[self.name_mapping[name]]] = self.tex2flow_mapping[self.name_mapping[name]]
                        self.inv_active_flow_classes_mapping[self.tex2flow_mapping[self.name_mapping[name]]] = self.tex2flow_mapping[self.name_mapping[name]]
        else:
            self.active_flow_classes_mapping = self.active_classes_mapping
            self.inv_active_flow_classes_mapping = self.inv_active_classes_mapping

        self.numClasses = len(self.classNames)

        if self.opt.isTrain:
            self.numFlowClasses = self.num_flow_classes if (opt.load_pretrained_flow != '' or opt.load_pretrain != '') else len(self.active_flow_classes_mapping)
        else:
            self.numFlowClasses = len(self.active_flow_classes_mapping) if ('single' in self.root) else self.num_flow_classes

        opt.numClasses = self.numClasses
        opt.classNames = self.classNames
        opt.numFlowClasses = self.numFlowClasses
        opt.active_classes_mapping = self.active_classes_mapping
        opt.tex2flow_mapping = self.tex2flow_mapping
        opt.inv_active_flow_classes_mapping = self.inv_active_flow_classes_mapping

        # set class counter for test mode
        if self.opt.isTrain is False:
            opt.batchSize = self.numClasses
            self.class_counter = 0
            self.img_counter = 0

        # arrange directories
        self.dirs = []
        self.img_paths = []
        if self.opt.use_parsings is True:
            self.parsing_paths = []
        self.sizes = []

        for currClass in self.classNames:
            self.dirs += [os.path.join(opt.dataroot, opt.phase + currClass)]
            imgs, parsings = make_dataset(self.dirs[-1], self.opt, None)
            self.img_paths += [imgs]
            self.parsing_paths += [parsings]
            self.sizes += [len(self.img_paths[-1])]

        opt.dataset_size = self.__len__()
        self.transform = get_transform(opt)

    def set_sample_mode(self, mode=False):
        self.get_samples = mode
        self.class_counter = 0
        self.img_counter = 0

    def assign_flow_class(self, class_name):
        if self.use_flow_classes:
            ages = [int(s) for s in re.split('-|_', class_name) if s.isdigit()]
            max_age = ages[-1]
            for i in range(len(self.flow_upper_bounds)):
                if max_age <= self.flow_upper_bounds[i]:
                    break

            return i
        else:
            return self.assign_tex_class(class_name)

    def assign_tex_class(self, class_name):
        ages = [int(s) for s in re.split('-|_', class_name) if s.isdigit()]
        max_age = ages[-1]
        for i in range(len(TEX_CLASSES_UPPER_BOUNDS)):
            if max_age <= TEX_CLASSES_UPPER_BOUNDS[i]:
                break

        return i

    def expand_parsing_labels(self, labels, num_out_labels=19):
        h, w, c = labels.shape
        max_label = num_out_labels - 1
        reshaped_labels = labels.reshape((1, -1))
        out = np.zeros((num_out_labels, h*w), dtype=np.uint8)
        if self.no_facial_hair:
            facial_hair_mask = np.zeros((3, h*w), dtype=np.uint8)
        if self.no_clothing_items:
            clothing_items_mask = np.zeros((3, h*w), dtype=np.uint8)

        for i in range(max_label + 1):
            ind = (reshaped_labels == i).reshape((h * w))
            if ((i >= 14 and i <= 16) or i == 18) and self.no_clothing_items:
                out[i, ind] = 255
                clothing_items_mask[:, ind] = 255
            else:
                out[i, ind] = 255

        parsing_out = np.transpose(out.reshape((num_out_labels, h, w)), (1, 2, 0))
        neck_out = None
        clothing_items_out = np.transpose(clothing_items_mask.reshape((3, h, w)), (1, 2, 0))
        facial_hair_out = np.transpose(facial_hair_mask.reshape((3, h, w)), (1, 2, 0))

        return parsing_out, facial_hair_out, clothing_items_out, neck_out

    def get_item_from_path(self, path):
        path_dir, im_name = os.path.split(path)

        try:
            img = Image.open(path + '.jpg').convert('RGB')
        except:
            img = Image.open(path + '.png').convert('RGB')

        img = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)
        if self.opt.use_parsings is True:
            parsing_path = os.path.join(path_dir, 'parsings', im_name + '.png')
            parsing = Image.open(parsing_path).convert('L')
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

        neck_mask = 0
        expanded_parsing = 0
        landmarks = 0

        return {'Imgs': img, 'facial_hair': facial_hair_mask,
                'expanded_parsings': expanded_parsing,
                'landmarks': landmarks,
                'clothing_items': clothing_items_mask, 'neck': neck_mask,
                'Paths': [path], 'Classes': torch.zeros(1,dtype=torch.int),
                'flow_Classes': torch.zeros(1, dtype=torch.int),
                'Valid': True}

    def __getitem__(self, index):
        if self.opt.isTrain and not self.get_samples:
            if self.mode == 'uniform_tex':
                # This is not really necessary
                # # make sure to not repeat the same translation twice in a row
                # self.prev_A, self.prev_B = self.class_A, self.class_B
                # while self.prev_A == self.class_A and self.prev_B == self.class_B:

                condition = True
                while condition:
                    self.class_A_idx, self.class_B_idx = np.random.permutation(self.numClasses)[:2]
                    # self.class_A_idx = random.randint(0,self.numClasses - 1)
                    self.class_A = self.active_classes_mapping[self.class_A_idx]
                    if self.use_flow_classes:
                        self.flow_class_A = self.tex2flow_mapping[self.class_A]
                    else:
                        self.flow_class_A = self.class_A
                    self.flow_class_A_idx = self.inv_active_flow_classes_mapping[self.flow_class_A]

                    # self.class_B_idx = random.randint(0,self.numClasses - 1)
                    self.class_B = self.active_classes_mapping[self.class_B_idx]
                    if self.use_flow_classes:
                        self.flow_class_B = self.tex2flow_mapping[self.class_B]
                    else:
                        self.flow_class_B = self.class_B
                    self.flow_class_B_idx = self.inv_active_flow_classes_mapping[self.flow_class_B]

                    if self.forward_pass_id_loss:
                        condition = self.class_A == self.class_B
                    else:
                        condition = False

                # if there is an identity transform load the same image to both classes
                # might not be necessary, it was like that in my original torch version though
                index_A = random.randint(0, self.sizes[self.class_A_idx] - 1)
                if self.class_A == self.class_B:
                    index_B = index_A
                else:
                    index_B = random.randint(0, self.sizes[self.class_B_idx] - 1)

            else: #self.mode == 'uniform_flow'
                # self.prev_A, self.prev_B = self.flow_class_A, self.flow_class_B
                # while self.prev_A == self.flow_class_A and self.prev_B == self.flow_class_B:

                condition = True
                while condition:
                    self.flow_class_A_idx, self.flow_class_B_idx = torch.randperm(self.numClasses)[:2]
                    # self.flow_class_A_idx = random.randint(0,self.numFlowClasses - 1)
                    self.flow_class_A = self.active_flow_classes_mapping[self.flow_class_A_idx]
                    possible_mappings_A = [i for i in self.flow2tex_mapping[self.flow_class_A] if i in self.active_classes_mapping.values()] #len(self.flow2tex_mapping[self.flow_class_A])
                    self.class_A = possible_mappings_A[random.randint(0,len(possible_mappings_A) - 1)]
                    self.class_A_idx = self.inv_active_classes_mapping[self.class_A]

                    # self.flow_class_B_idx = random.randint(0,self.numFlowClasses - 1)
                    self.flow_class_B = self.active_flow_classes_mapping[self.flow_class_B_idx]
                    possible_mappings_B = [i for i in self.flow2tex_mapping[self.flow_class_B] if i in self.active_classes_mapping.values()] #len(self.flow2tex_mapping[self.flow_class_B])
                    self.class_B = possible_mappings_B[random.randint(0,len(possible_mappings_B) - 1)]
                    self.class_B_idx = self.inv_active_classes_mapping[self.class_B]

                    # if there is an identity transform load the same image to both classes
                    # might not be necessary, it was like that in my original torch version though
                    index_A = random.randint(0, self.sizes[self.class_A_idx] - 1)
                    if self.flow_class_A == self.flow_class_B:
                        self.class_B = self.class_A
                        self.class_B_idx = self.class_A_idx
                        index_B = index_A
                    else:
                        index_B = random.randint(0, self.sizes[self.class_B_idx] - 1)

                    if self.forward_pass_id_loss:
                        condition = self.flow_class_A == self.flow_class_B
                    else:
                        condition = False

            A_img_path = self.img_paths[self.class_A_idx][index_A]
            A_img = Image.open(A_img_path).convert('RGB')
            A_img = np.array(A_img.getdata(), dtype=np.uint8).reshape(A_img.size[1], A_img.size[0], 3)
            B_img_path = self.img_paths[self.class_B_idx][index_B]
            B_img = Image.open(B_img_path).convert('RGB')
            B_img = np.array(B_img.getdata(), dtype=np.uint8).reshape(B_img.size[1], B_img.size[0], 3)
            if self.opt.use_parsings is True:
                A_parsing_path = self.parsing_paths[self.class_A_idx][index_A]
                A_parsing = Image.open(A_parsing_path).convert('L')
                A_parsing = np.array(A_parsing.getdata(), dtype=np.uint8).reshape(A_parsing.size[1], A_parsing.size[0], 1)
                expanded_A_parsing, facial_hair_mask_A, clothing_items_mask_A, neck_mask_A = self.expand_parsing_labels(A_parsing, self.parsing_labels_num)
                A_parsing_img = util.parsingLabels2image(expanded_A_parsing/255)
                A_img = np.concatenate((A_img, A_parsing_img), axis=2)

                B_parsing_path = self.parsing_paths[self.class_B_idx][index_B]
                B_parsing = Image.open(B_parsing_path).convert('L')
                B_parsing = np.array(B_parsing.getdata(), dtype=np.uint8).reshape(B_parsing.size[1], B_parsing.size[0], 1)
                expanded_B_parsing, facial_hair_mask_B, clothing_items_mask_B, neck_mask_B = self.expand_parsing_labels(B_parsing, self.parsing_labels_num)
                B_parsing_img = util.parsingLabels2image(expanded_B_parsing/255)
                B_img = np.concatenate((B_img, B_parsing_img), axis=2)

            # numpy conversions are an annoying hack to form a PIL image with more than 3 channels
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
            if self.no_facial_hair:
                facial_hair_mask_A = self.transform(facial_hair_mask_A)
                facial_hair_mask_B = self.transform(facial_hair_mask_B)
            else:
                facial_hair_mask_A = 0
                facial_hair_mask_B = 0

            if self.no_clothing_items:
                clothing_items_mask_A = self.transform(clothing_items_mask_A)
                clothing_items_mask_B = self.transform(clothing_items_mask_B)
            else:
                clothing_items_mask_A = 0
                clothing_items_mask_B = 0

            neck_mask_A = 0
            neck_mask_B = 0
            expanded_A_parsing = 0
            expanded_B_parsing = 0
            A_landmarks = 0
            B_landmarks = 0

            return {'A': A_img, 'B': B_img,
                    'facial_hair_A': facial_hair_mask_A, 'facial_hair_B': facial_hair_mask_B,
                    'expanded_A_parsing': expanded_A_parsing, 'expanded_B_parsing': expanded_B_parsing,
                    'landmarks_A': A_landmarks, 'landmarks_B': B_landmarks,
                    'clothing_items_A': clothing_items_mask_A, 'clothing_items_B': clothing_items_mask_B,
                    'neck_A': neck_mask_A, 'neck_B': neck_mask_B,
                    "A_class": self.class_A_idx, "B_class": self.class_B_idx,
                    "flow_A_class": self.flow_class_A_idx, "flow_B_class": self.flow_class_B_idx,
                    'A_paths': A_img_path, 'B_paths': B_img_path}

        else:  # in test mode, load one image from each class
            i = self.class_counter % self.numClasses
            if self.use_flow_classes:
                flow_class = self.tex2flow_mapping[self.active_classes_mapping[i]]
            else:
                flow_class = self.active_classes_mapping[i]

            flow_class_idx = self.inv_active_flow_classes_mapping[flow_class]
            self.class_counter += 1
            if self.get_samples:
                ind = random.randint(0, self.sizes[i] - 1)
            else:
                ind = self.img_counter if self.img_counter < self.sizes[i] else -1
            if i == self.numClasses - 1:
                self.img_counter += 1

            if ind > -1:
                valid = True
                paths = self.img_paths[i][ind]
                img = Image.open(self.img_paths[i][ind]).convert('RGB')
                img = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)
                if self.opt.use_parsings is True:
                    parsing_path = self.parsing_paths[i][ind]
                    parsing = Image.open(parsing_path).convert('L')
                    parsing = np.array(parsing.getdata(), dtype=np.uint8).reshape(parsing.size[1], parsing.size[0], 1)
                    expanded_parsing, facial_hair_mask, clothing_items_mask, neck_mask = self.expand_parsing_labels(parsing, self.parsing_labels_num)
                    parsing_img = util.parsingLabels2image(expanded_parsing/255)
                    img = np.concatenate((img, parsing_img), axis=2)

                img = self.transform(img)
                if self.no_facial_hair:
                    facial_hair_mask = self.transform(facial_hair_mask)
                else:
                    facial_hair_mask = 0

                if self.no_clothing_items:
                    clothing_items_mask = self.transform(clothing_items_mask)
                else:
                    clothing_items_mask = 0

                neck_mask = 0
                expanded_parsing = 0
                landmarks = 0

            else:
                if self.opt.use_parsings:
                    img = torch.zeros(6, self.opt.fineSize, self.opt.fineSize)
                else:
                    img = torch.zeros(3, self.opt.fineSize, self.opt.fineSize)

                if self.no_facial_hair:
                    facial_hair_mask = torch.zeros(3, self.opt.fineSize, self.opt.fineSize)
                else:
                    facial_hair_mask = 0

                if self.no_clothing_items:
                    clothing_items_mask = torch.zeros(3, self.opt.fineSize, self.opt.fineSize)
                else:
                    clothing_items_mask = 0

                neck_mask = 0
                expanded_parsing = 0
                landmarks = 0

                paths = ''
                valid = False

            return {'Imgs': img, 'facial_hair': facial_hair_mask,
                    'expanded_parsings': expanded_parsing,
                    'landmarks': landmarks,
                    'clothing_items': clothing_items_mask, 'neck': neck_mask,
                    'Paths': paths, 'Classes': i, 'flow_Classes': flow_class_idx,
                    'Valid': valid}


    def __len__(self):
        if self.opt.isTrain:
            return round(sum(self.sizes) / 2)  # this determines how many iterations we make per epoch
        else:
            return max(self.sizes) * self.numClasses

    def name(self):
        return 'MulticlassUnalignedDataset'
