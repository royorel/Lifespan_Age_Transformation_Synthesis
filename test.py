### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import scipy # this is to prevent a potential error caused by importing torch before scipy (happens due to a bad combination of torch & scipy versions)
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from pdb import set_trace as st


def test(opt):
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#test batches = %d' % (int(dataset_size / len(opt.sort_order))))
    visualizer = Visualizer(opt)
    model = create_model(opt)
    model.eval()

    # create webpage
    if opt.random_seed != -1:
        exp_dir = '%s_%s_seed%s' % (opt.phase, opt.which_epoch, str(opt.random_seed))
    else:
        exp_dir = '%s_%s' % (opt.phase, opt.which_epoch)
    web_dir = os.path.join(opt.results_dir, opt.name, exp_dir)

    if opt.traverse or opt.deploy:
        if opt.traverse:
            out_dirname = 'traversal'
        else:
            out_dirname = 'deploy'
        output_dir = os.path.join(web_dir,out_dirname)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for image_path in opt.image_path_list:
            print(image_path)
            data = dataset.dataset.get_item_from_path(image_path)
            visuals = model.inference(data)
            if opt.traverse and opt.make_video:
                out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.mp4')
                visualizer.make_video(visuals, out_path)
            elif opt.traverse or (opt.deploy and opt.full_progression):
                if opt.traverse and opt.compare_to_trained_outputs:
                    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_compare_to_{}_jump_{}.png'.format(opt.compare_to_trained_class, opt.trained_class_jump))
                else:
                    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
                visualizer.save_row_image(visuals, out_path, traverse=opt.traverse)
            else:
                out_path = os.path.join(output_dir, os.path.basename(image_path[:-4]))
                visualizer.save_images_deploy(visuals, out_path)
    else:
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

        # test
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break

            visuals = model.inference(data)
            img_path = data['Paths']
            rem_ind = []
            for i, path in enumerate(img_path):
                if path != '':
                    print('process image... %s' % path)
                else:
                    rem_ind += [i]

            for ind in reversed(rem_ind):
                del img_path[ind]

            visualizer.save_images(webpage, visuals, img_path)

            webpage.save()


if __name__ == "__main__":
    opt = TestOptions().parse(save=False)
    test(opt)
