### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import scipy # this is to prevent a potential error caused by importing torch before scipy (happens due to a bad combination of torch & scipy versions)
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from pdb import set_trace as st

def train(opt):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    if opt.continue_train:
        if opt.which_epoch == 'latest':
            try:
                start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
            except:
                start_epoch, epoch_iter = 1, 0
        else:
            start_epoch, epoch_iter = int(opt.which_epoch), 0

        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
        for update_point in opt.decay_epochs:
            if start_epoch < update_point:
                break

            opt.lr *= opt.decay_gamma
    else:
        start_epoch, epoch_iter = 0, 0

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)

    total_steps = (start_epoch) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    bSize = opt.batchSize

    #in case there's no display sample one image from each class to test after every epoch
    if opt.display_id == 0:
        dataset.dataset.set_sample_mode(True)
        dataset.num_workers = 1
        for i, data in enumerate(dataset):
            if i*opt.batchSize >= opt.numClasses:
                break
            if i == 0:
                sample_data = data
            else:
                for key, value in data.items():
                    if torch.is_tensor(data[key]):
                        sample_data[key] = torch.cat((sample_data[key], data[key]), 0)
                    else:
                        sample_data[key] = sample_data[key] + data[key]
        dataset.num_workers = opt.nThreads
        dataset.dataset.set_sample_mode(False)

    for epoch in range(start_epoch, opt.epochs):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = 0
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = (total_steps % opt.display_freq == display_delta) and (opt.display_id > 0)

            ############## Network Pass ########################
            model.set_inputs(data)
            disc_losses = model.update_D()
            gen_losses, gen_in, gen_out, rec_out, cyc_out = model.update_G(infer=save_fake)
            loss_dict = dict(gen_losses, **disc_losses)
            ##################################################

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.item() if not (isinstance(v, float) or isinstance(v, int)) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch+1, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            ### display output images
            if save_fake and opt.display_id > 0:
                class_a_suffix = ' class {}'.format(data['A_class'][0])
                class_b_suffix = ' class {}'.format(data['B_class'][0])
                classes = None

                visuals = OrderedDict()
                visuals_A = OrderedDict([('real image' + class_a_suffix, util.tensor2im(gen_in.data[0]))])
                visuals_B = OrderedDict([('real image' + class_b_suffix, util.tensor2im(gen_in.data[bSize]))])

                A_out_vis = OrderedDict([('synthesized image' + class_b_suffix, util.tensor2im(gen_out.data[0]))])
                B_out_vis = OrderedDict([('synthesized image' + class_a_suffix, util.tensor2im(gen_out.data[bSize]))])
                if opt.lambda_rec > 0:
                    A_out_vis.update([('reconstructed image' + class_a_suffix, util.tensor2im(rec_out.data[0]))])
                    B_out_vis.update([('reconstructed image' + class_b_suffix, util.tensor2im(rec_out.data[bSize]))])
                if opt.lambda_cyc > 0:
                    A_out_vis.update([('cycled image' + class_a_suffix, util.tensor2im(cyc_out.data[0]))])
                    B_out_vis.update([('cycled image' + class_b_suffix, util.tensor2im(cyc_out.data[bSize]))])

                visuals_A.update(A_out_vis)
                visuals_B.update(B_out_vis)
                visuals.update(visuals_A)
                visuals.update(visuals_B)

                ncols = len(visuals_A)
                visualizer.display_current_results(visuals, epoch, classes, ncols)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch+1, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
                if opt.display_id == 0:
                    model.eval()
                    visuals = model.inference(sample_data)
                    visualizer.save_matrix_image(visuals, 'latest')
                    model.train()

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch+1, opt.epochs, time.time() - epoch_start_time))

        ### save model for this epoch
        if (epoch+1) % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch+1, total_steps))
            model.save('latest')
            model.save(epoch+1)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
            if opt.display_id == 0:
                model.eval()
                visuals = model.inference(sample_data)
                visualizer.save_matrix_image(visuals, epoch+1)
                model.train()

        ### multiply learning rate by opt.decay_gamma after certain iterations
        if (epoch+1) in opt.decay_epochs:
            model.update_learning_rate()

if __name__ == "__main__":
    opt = TrainOptions().parse()
    train(opt)
