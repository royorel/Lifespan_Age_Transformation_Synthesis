import os
import dlib
import shutil
import requests
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms as transforms
import util.deeplab as deeplab
from PIL import Image
from util.util import download_file
from pdb import set_trace as st

resnet_file_path = 'deeplab_model/R-101-GN-WS.pth.tar'
deeplab_file_path = 'deeplab_model/deeplab_model.pth'
predictor_file_path = 'util/shape_predictor_68_face_landmarks.dat'
model_fname = 'deeplab_model/deeplab_model.pth'
deeplab_classes = ['background' ,'skin','nose','eye_g','l_eye','r_eye','l_brow','r_brow','l_ear','r_ear','mouth','u_lip','l_lip','hair','hat','ear_r','neck_l','neck','cloth']


class preprocessInTheWildImage():
    def __init__(self, out_size=256):
        self.out_size = out_size

        # load landmark detector models
        self.detector = dlib.get_frontal_face_detector()
        if not os.path.isfile(predictor_file_path):
            print('Cannot find landmarks shape predictor model.\n'\
                  'Please run download_models.py to download the model')
            raise OSError

        self.predictor = dlib.shape_predictor(predictor_file_path)

        # deeplab data properties
        self.deeplab_data_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.deeplab_input_size = 513

        # load deeplab model
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        if not os.path.isfile(resnet_file_path):
            print('Cannot find DeeplabV3 backbone Resnet model.\n' \
                  'Please run download_models.py to download the model')
            raise OSError

        self.deeplab_model = getattr(deeplab, 'resnet101')(
        	                       pretrained=True,
        	                       num_classes=len(deeplab_classes),
        	                       num_groups=32,
        	                       weight_std=True,
        	                       beta=False)

        self.deeplab_model.eval()
        if not os.path.isfile(deeplab_file_path):
            print('Cannot find DeeplabV3 model.\n' \
                  'Please run download_models.py to download the model')
            raise OSError

        checkpoint = torch.load(model_fname)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        self.deeplab_model.load_state_dict(state_dict)

    def dlib_shape_to_landmarks(self, shape):
    	# initialize the list of (x, y)-coordinates
    	landmarks = np.zeros((68, 2), dtype=np.float32)
    	# loop over the 68 facial landmarks and convert them
    	# to a 2-tuple of (x, y)-coordinates
    	for i in range(0, 68):
    		landmarks[i] = (shape.part(i).x, shape.part(i).y)
    	# return the list of (x, y)-coordinates
    	return landmarks

    def extract_face_landmarks(self, img):
        # detect all faces in the image and
        # keep the detection with the largest bounding box
        dets = self.detector(img, 1)
        if len(dets) == 0:
            print ('Could not detect any face in the image, please try again with a different image')
            raise

        max_area = 0
        max_idx = -1
        for k, d in enumerate(dets):
            area = (d.right() - d.left()) * (d.bottom() - d.top())
            if area > max_area:
                max_area = area
                max_idx = k

        # Get the landmarks/parts for the face in box d.
        dlib_shape = self.predictor(img, dets[max_idx])
        landmarks = self.dlib_shape_to_landmarks(dlib_shape)
        return landmarks

    def align_in_the_wild_image(self, np_img, lm, transform_size=4096, enable_padding=True):
        # Parse landmarks.
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 2.2) # This results in larger crops then the original FFHQ. For the original crops, replace 2.2 with 1.8
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        img = Image.fromarray(np_img)

        # Shrink.
        shrink = int(np.floor(qsize / self.out_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        if self.out_size < transform_size:
            img = img.resize((self.out_size, self.out_size), Image.ANTIALIAS)

        return img


    def get_segmentation_maps(self, img):
        img = img.resize((self.deeplab_input_size,self.deeplab_input_size),Image.BILINEAR)
        img = self.deeplab_data_transform(img)
        img = img.cuda()
        self.deeplab_model.cuda()
        outputs = self.deeplab_model(img.unsqueeze(0))
        self.deeplab_model.cpu()
        _, pred = torch.max(outputs, 1)
        pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
        seg_map = Image.fromarray(pred)
        seg_map = np.uint8(seg_map.resize((self.out_size,self.out_size), Image.NEAREST))
        return seg_map

    def forward(self, img):
        landmarks = self.extract_face_landmarks(img)
        aligned_img = self.align_in_the_wild_image(img, landmarks)
        seg_map = self.get_segmentation_maps(aligned_img)
        aligned_img = np.array(aligned_img.getdata(), dtype=np.uint8).reshape(self.out_size, self.out_size, 3)
        return aligned_img, seg_map
