### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import PIL.Image as Image
import os
import csv
import shutil
from pdb import set_trace as st

clusters = ['0-2','3-6','7-9','10-14','15-19',
            '20-29','30-39','40-49','50-69','70-120']

def processIm(img_filename, phase, csv_row, num):
    img_basename = os.path.basename(img_filename)
    labels_filename = os.path.join(os.path.dirname(img_filename), 'parsings', img_basename)

    age, age_conf = csv_row['age_group'], float(csv_row['age_group_confidence'])
    gender, gender_conf = csv_row['gender'], float(csv_row['gender_confidence'])
    head_pitch, head_roll, head_yaw = float(csv_row['head_pitch']), float(csv_row['head_roll']), float(csv_row['head_yaw'])
    left_eye_occluded, right_eye_occluded = float(csv_row['left_eye_occluded']), float(csv_row['right_eye_occluded'])
    glasses = csv_row['glasses']

    no_attributes_found = head_pitch == -1 and head_roll == -1 and head_yaw == -1 and \
                          left_eye_occluded == -1 and right_eye_occluded == -1 and glasses == -1

    age_cond = age_conf > 0.6
    gender_cond = gender_conf > 0.66
    head_pose_cond = abs(head_pitch) < 30.0 and abs(head_yaw) < 40.0
    eyes_cond = (left_eye_occluded < 90.0 and right_eye_occluded < 50.0) or (left_eye_occluded < 50.0 and right_eye_occluded < 90.0)
    glasses_cond = glasses != 'Dark'

    valid1 = age_cond and gender_cond and no_attributes_found
    valid2 = age_cond and gender_cond and head_pose_cond and eyes_cond and glasses_cond

    if gender == 'male':
        dst_gender = 'males'
    else:
        dst_gender = 'females'

    dst_cluster = phase + age

    if (valid1 or valid2):
        dst_path = os.path.join(dst_gender, dst_cluster, img_basename)
        parsing_dst_path = os.path.join(dst_gender, dst_cluster, 'parsings', img_basename)
        shutil.copy(img_filename, dst_path)
        shutil.copy(labels_filename, parsing_dst_path)


def create_dataset(folder, labels_file, train_split):
    # create all necessary subfolders
    for clust in clusters:
        trainMaleClusterPath = "males/train" + clust
        testMaleClusterPath = "males/test" + clust
        trainFemaleClusterPath = "females/train" + clust
        testFemleClusterPath = "females/test" + clust

        if not os.path.isdir(trainMaleClusterPath):
            os.makedirs(trainMaleClusterPath)
            os.makedirs(os.path.join(trainMaleClusterPath,'parsings'))
            os.makedirs(testMaleClusterPath)
            os.makedirs(os.path.join(testMaleClusterPath,'parsings'))
            os.makedirs(trainFemaleClusterPath)
            os.makedirs(os.path.join(trainFemaleClusterPath,'parsings'))
            os.makedirs(testFemleClusterPath)
            os.makedirs(os.path.join(testFemleClusterPath,'parsings'))

    # process images
    with open(labels_file,'r', newline='') as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            num = int(csv_row['image_number'])

            if num < train_split:
                phase = 'train'
            else:
                phase = 'test'

            subdir = str(num - (num % 1000)).zfill(5)
            img_filename = os.path.join(folder,subdir,str(num).zfill(5)+'.png')
            if os.path.isfile(img_filename):
                print('processing {}'.format(img_filename))
                processIm(img_filename, phase, csv_row, num)
            else:
                print('Image {}.png was not found'.format(str(num).zfill(5)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--folder', type=str, default='../../FFHQ-Aging-Dataset/ffhq_aging256x256', help='Location of the raw FFHQ-Aging dataset')
    argparser.add_argument('--labels_file', type=str, default='../../FFHQ-Aging-Dataset/ffhq_aging_labels.csv', help='Location of the raw FFHQ-Aging dataset')
    argparser.add_argument('--train_split', type=int, default=69000, help='number of images to allocate for training')
    args = argparser.parse_args()
    folder = args.folder
    labels_file = args.labels_file
    train_split = args.train_split
    create_dataset(folder, labels_file, train_split)
