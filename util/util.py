### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import html
import glob
import uuid
import hashlib
import requests
import torch
import zipfile
import numpy as np
from tqdm import tqdm
from PIL import Image
from pdb import set_trace as st


males_model_spec = dict(file_url='https://drive.google.com/uc?id=1MsXN54hPi9PWDmn1HKdmKfv-J5hWYFVZ',
                        alt_url='https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/pretrained_models/males_model.zip',
                        file_path='checkpoints/males_model.zip', file_size=213175683, file_md5='0079186147ec816176b946a073d1f396')
females_model_spec = dict(file_url='https://drive.google.com/uc?id=1LNm0zAuiY0CIJnI0lHTq1Ttcu9_M1NAJ',
                          alt_url='https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/pretrained_models/females_model.zip',
                          file_path='checkpoints/females_model.zip', file_size=213218113, file_md5='0675f809413c026170cf1f22b27f3c5d')
resnet_file_spec = dict(file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM',
                        alt_url='https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/pretrained_models/R-101-GN-WS.pth.tar',
                        file_path='deeplab_model/R-101-GN-WS.pth.tar', file_size=178260167, file_md5='aa48cc3d3ba3b7ac357c1489b169eb32')
deeplab_file_spec = dict(file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY',
                         alt_url='https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/pretrained_models/deeplab_model.pth',
                         file_path='deeplab_model/deeplab_model.pth', file_size=464446305, file_md5='8e8345b1b9d95e02780f9bed76cc0293')
predictor_file_spec = dict(file_url='https://drive.google.com/uc?id=1fhq5lvWy-rjrzuHdMoZfLsULvF0gJGwD',
                           alt_url='https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/pretrained_models/shape_predictor_68_face_landmarks.dat',
                           file_path='util/shape_predictor_68_face_landmarks.dat', file_size=99693937, file_md5='73fde5e05226548677a050913eed4e04')

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

def download_pretrained_models():
    print('Downloading males model')
    with requests.Session() as session:
        try:
            download_file(session, males_model_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, males_model_spec, use_alt_url=True)

    print('Extracting males model zip file')
    with zipfile.ZipFile('./checkpoints/males_model.zip','r') as zip_fname:
        zip_fname.extractall('./checkpoints')

    print('Done!')
    os.remove(males_model_spec['file_path'])

    print('Downloading females model')
    with requests.Session() as session:
        try:
            download_file(session, females_model_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, females_model_spec, use_alt_url=True)

    print('Extracting females model zip file')
    with zipfile.ZipFile('./checkpoints/females_model.zip','r') as zip_fname:
        zip_fname.extractall('./checkpoints')

    print('Done!')
    os.remove(females_model_spec['file_path'])

    print('Downloading face landmarks shape predictor')
    with requests.Session() as session:
        try:
            download_file(session, predictor_file_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, predictor_file_spec, use_alt_url=True)

    print('Done!')

    print('Downloading DeeplabV3 backbone Resnet Model parameters')
    with requests.Session() as session:
        try:
            download_file(session, resnet_file_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, resnet_file_spec, use_alt_url=True)

    print('Done!')

    print('Downloading DeeplabV3 Model parameters')
    with requests.Session() as session:
        try:
            download_file(session, deeplab_file_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, deeplab_file_spec, use_alt_url=True)

    print('Done!')

def download_file(session, file_spec, use_alt_url=False, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    if use_alt_url:
        file_url = file_spec['alt_url']
    else:
        file_url = file_spec['file_url']

    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    progress_bar = tqdm(total=file_spec['file_size'], unit='B', unit_scale=True)
    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        progress_bar.reset()
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        progress_bar.update(len(chunk))
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

    progress_bar.close()

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass
