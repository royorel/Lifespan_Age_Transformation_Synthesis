### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from pdb import set_trace as st

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def list_folder_images(dir, opt):
    images = []
    parsings = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in os.listdir(dir):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            # make sure there's a matching parsings for the image
            # parsing files must be png
            parsing_fname = fname[:-3] + 'png'
            if os.path.isfile(os.path.join(dir, 'parsings', parsing_fname)):
                parsing_path = os.path.join(dir, 'parsings', parsing_fname)
                images.append(path)
                parsings.append(parsing_path)

    # sort according to identity in case of FGNET test
    if 'fgnet' in opt.dataroot.lower():
        images.sort(key=str.lower)
        parsings.sort(key=str.lower)

    return images, parsings
