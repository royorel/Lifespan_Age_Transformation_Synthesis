### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import html
import glob
import uuid
import hashlib
import requests
import torch
import numpy as np
from PIL import Image
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

def download_file(session, file_spec, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    file_url = file_spec['file_url']
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            break

        except:
            # Last attempt => raise error.
            if not attempts_left:
                raise

            # Handle Google Drive virus checker nag.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                links = [html.unescape(link) for link in data.decode('utf-8').split('"') if 'export=download' in link]
                if len(links) == 1:
                    file_url = requests.compat.urljoin(file_url, links[0])
                    continue

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass
