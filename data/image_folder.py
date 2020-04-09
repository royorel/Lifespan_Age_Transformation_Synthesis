###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories.
###############################################################################
# Roy Or-El 10/4/2017
# Modified the code to consider cropped, masks and parsings subdirectories
# - If use_cropped is True, only images are loaded from a subdirectory named "cropped" if such directory exists.
#   Otherwise, it is assumed that the files in the main directory are cropped and only loads them
# - If use_masks or use_parsings is True, make_dataset will also return path lists for the masks and/or parsings.
#   In addition, only images from the main directory are considered to avoid conflicts with the masks and parsings
#   subdirectories.
###############################################################################

import torch.utils.data as data
from pdb import set_trace as st
from PIL import Image
import os
import os.path
import re

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.json',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, opt, frontal_dict=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    if opt.frontal_only and frontal_dict is None:
        print("No frontal face information for this dataset, loading all images!")

    if opt.use_masks is False and opt.use_parsings is False and opt.use_cropped is False:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    if opt.frontal_only and frontal_dict is not None:
                        backslashes = [m for m in re.finditer('/', path)]
                        frontal_dict_key = path[backslashes[-2].span()[1]:]
                        if frontal_dict[frontal_dict_key]:
                            images.append(path)
                    else:
                        images.append(path)
    elif opt.use_cropped is True and os.path.isdir(os.path.join(dir, "cropped")):
        for fname in os.listdir(os.path.join(dir, "cropped")):
            if is_image_file(fname):
                path = os.path.join(dir, "cropped", fname)
                if opt.frontal_only and frontal_dict is not None:
                    backslashes = [m for m in re.finditer('/', path)]
                    frontal_dict_key = path[backslashes[-2].span()[1]:]
                    if frontal_dict[frontal_dict_key]:
                        images.append(path)
                else:
                    images.append(path)
    else:
        for fname in os.listdir(dir):
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                if opt.frontal_only and frontal_dict is not None:
                    backslashes = [m for m in re.finditer('/', path)]
                    frontal_dict_key = path[backslashes[-2].span()[1]:]
                    if frontal_dict[frontal_dict_key]:
                        images.append(path)
                else:
                    images.append(path)

    # sort according to identity in case of FGNET test
    if 'fgnet' in opt.dataroot.lower() or 'youtube' in opt.dataroot.lower():
        images.sort(key=str.lower)

    if opt.use_masks is True:
        masks = []
        no_mask_ind = []
        if os.path.isdir(os.path.join(dir, "masks", "cropped")):
            masks_dir = os.path.join("masks","cropped")
            mask_suffix = "_mask_cropped.png"
        elif os.path.isdir(os.path.join(dir,"masks_aaron")):
            masks_dir = "masks_aaron"
            mask_suffix = ""
        else:
            masks_dir = "masks"
            mask_suffix = "_mask.png"

        mask_files = os.listdir(os.path.join(dir, masks_dir))
        # add masks
        for i, fname in enumerate(images):
            basename = os.path.basename(fname)
            if "cropped" in fname.lower():
                last_underscore = basename.rfind("_")
                imName = basename[:last_underscore]
            else:
                last_dot = basename.rfind(".")
                imName = basename[:last_dot]

            mask_ind = [j for j in range(len(mask_files)) if imName + mask_suffix in mask_files[j]]
            if len(mask_ind) > 0:
                path = os.path.join(dir, masks_dir, mask_files[mask_ind[0]])
                masks.append(path)
            else:
                no_mask_ind += [i]

        # remove images that has no masks
        # start from the highest index to avoid deleting wrong entries
        for ind in reversed(no_mask_ind):
            del images[ind]

    if opt.use_landmarks is True:
        landmarks = []
        no_landmarks_ind = []
        landmarks_suffix = ""
        if opt.embed_landmarks or 'nvidia' in dir:
            landmarks_dir = 'landmarks'
        else:
            landmarks_dir = 'color_landmarks'

        landmarks_files = os.listdir(os.path.join(dir, landmarks_dir))
        # add landmarks
        for i, fname in enumerate(images):
            basename = os.path.basename(fname)
            last_dot = basename.rfind(".")
            imName = basename[:last_dot]

            landmarks_ind = [j for j in range(len(landmarks_files)) if imName + landmarks_suffix in landmarks_files[j]]
            if len(landmarks_ind) > 0:
                path = os.path.join(dir, landmarks_dir, landmarks_files[landmarks_ind[0]])
                landmarks.append(path)
            else:
                no_landmarks_ind += [i]

        # remove images that have no masks
        # start from the highest index to avoid deleting wrong entries
        for ind in reversed(no_landmarks_ind):
            del images[ind]
            if opt.use_masks is True:
                del masks[ind]

    if opt.use_parsings is True:
        parsing_format = opt.parsing_format
        parsings = []
        no_parsing_ind = []
        if parsing_format == 'image' and opt.use_hair_parsing is True:
            if os.path.isdir(os.path.join(dir, "parsings", "hair", "cropped")):
                parsings_dir = os.path.join("parsings","hair","cropped")
                parsings_suffix = "_parsed_cropped.jpg"
            else:
                parsings_dir = os.path.join("parsings","hair")
                if os.path.isdir(os.path.join(dir, "cropped")):
                    parsings_suffix = "_cropped_parsed.jpg"
                else:
                    parsings_suffix = "_parsed.jpg"
        elif parsing_format == 'image':
            if os.path.isdir(os.path.join(dir, "parsings", "no_hair", "cropped")):
                parsings_dir = os.path.join("parsings","no_hair","cropped")
                parsings_suffix = "_parsed_no_hair_cropped.jpg"
            else:
                parsings_dir = os.path.join("parsings","no_hair")
                if os.path.isdir(os.path.join(dir, "cropped")):
                    parsings_suffix = "_cropped_parsed_no_hair.jpg"
                else:
                    parsings_suffix = "_parsed_no_hair.jpg"
        else: # parsing format is labels
            if os.path.isdir(os.path.join(dir, "parsings", "labels", "cropped")):
                parsings_dir = os.path.join("parsings","labels","cropped")
                parsings_suffix = "_parsed_labels_cropped.png"
            elif os.path.isdir(os.path.join(dir, "parsings", "labels")):
                parsings_dir = os.path.join("parsings","labels")
                parsings_suffix = "_parsed_labels.png"
            else:
                parsings_dir = "parsings"
                parsings_suffix = ".png"
                # if ('final_face_parsing_dataset' in dir) or ('FGNET_with_pix2pixHD_labels' in dir) or ('single' in dir) or ('nvidia' in dir) or ('youtube' in dir):
                #     parsings_suffix = ".png"
                # else:
                #     parsings_suffix = "_parsed_labels.png"

        parsings_files = os.listdir(os.path.join(dir, parsings_dir))
        # add masks
        for i, fname in enumerate(images):
            basename = os.path.basename(fname)
            if "cropped" in fname.lower():
                last_underscore = basename.rfind("_")
                imName = basename[:last_underscore]
            else:
                last_dot = basename.rfind(".")
                imName = basename[:last_dot]

            parsings_ind = [j for j in range(len(parsings_files)) if imName + parsings_suffix in parsings_files[j]]
            if len(parsings_ind) > 0:
                path = os.path.join(dir, parsings_dir, parsings_files[parsings_ind[0]])
                parsings.append(path)
            else:
                parsings_ind = [j for j in range(len(parsings_files)) if imName + '_aligned' + parsings_suffix in parsings_files[j]]
                if len(parsings_ind) > 0:
                    # path = os.path.join(dir, parsings_dir, imName + parsings_suffix)
                    path = os.path.join(dir, parsings_dir, parsings_files[parsings_ind[0]])
                    parsings.append(path)
                else:
                    no_parsing_ind += [i]

        # remove images that has no masks
        # start from the highest index to avoid deleting wrong entries
        for ind in reversed(no_parsing_ind):
            del images[ind]
            if opt.use_masks is True:
                del masks[ind]
            if opt.use_landmarks is True:
                del landmarks[ind]

    if opt.use_masks is True and opt.use_parsings is True and opt.use_landmarks is True:
        return images, masks, parsings, landmarks
    elif opt.use_masks is True and opt.use_parsings is True:
        return images, masks, parsings
    elif opt.use_masks is True and opt.use_landmarks is True:
        return images, masks, landmarks
    elif opt.use_parsings is True and opt.use_landmarks is True:
        return images, parsings, landmarks
    elif opt.use_masks is True:
        return images, masks
    elif opt.use_parsings is True:
        return images, parsings
    elif opt.use_landmarks is True:
        return images, landmarks
    else:
        return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
