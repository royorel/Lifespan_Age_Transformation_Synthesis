### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
from pdb import set_trace as st

def train(opt):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    use_tensorboard = False

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

    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

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

    first_epoch_delta = 1
    first_epoch_save_freq = dataset_size // 10

    loader_mode = model.set_loader_mode()
    data_loader.dataset.mode = loader_mode

    # initialize generators to produce identity transform
    # not_warmup = 0 if opt.gan_mode != 'flow_only' else 1
    not_warmup = 1 #if opt.continue_train else 0
    texture_train = 1 if opt.gan_mode != 'flow_only' else 0
    if opt.add_epoches:
        finish_epoch = start_epoch + opt.niter + opt.niter_decay
    else:
        finish_epoch = opt.niter + opt.niter_decay

    # set data conversion function
    if opt.netG == 'adain_gen' and opt.parsings_transformation and opt.use_expanded_parsings:
        conversion_func = util.parsingLabels2image
    else:
        conversion_func = util.tensor2im

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

    for epoch in range(start_epoch, finish_epoch):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            # epoch_iter = epoch_iter % dataset_size
            epoch_iter = 0
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # stop warming up
            if total_steps > dataset_size:
                not_warmup = 1

            # whether to collect output images
            save_fake = (total_steps % opt.display_freq == display_delta) and (opt.display_id > 0)

            ############## MUNIT Pass ########################
            model.encode_input(data)
            disc_losses = model.update_D()
            gen_losses, gen_in, gen_flow, gen_parsing, opt_flow, gen_tex, rec_tex, cyc_tex, gen_seg, rec_seg, cyc_seg = model.update_G(infer=save_fake)
            loss_dict = dict(gen_losses, **disc_losses)
            ##################################################


            ############## Forward Pass ######################
            # if ((i + 1) % 5) == 0:
            #     optG_wgan = True
            # else:
            #     optG_wgan = False
            #
            # losses, gen_flow, gen_tex, gen_tex_parsings, gen_tex, rec_tex_parsings, rec_flow, \
            # gen_parsing, rec_parsing, gen_landmarks, rec_landmarks, opt_flow, inv_opt_flow = model(data, infer=save_fake, optG_wgan=optG_wgan)
            #
            # # sum per device losses
            # losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            # loss_dict = dict(zip(model.loss_names, losses))
            # if 'FlowTV' in loss_dict.keys():
            #     loss_dict['FlowTV'] = loss_dict['FlowTV'] * not_warmup
            #
            # # calculate final loss scalar
            # loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 * not_warmup + \
            #          (loss_dict['D_fake_class'] + loss_dict['D_real_class'] + loss_dict.get('Grad_penalty',0)) * not_warmup
            #
            # loss_G = loss_dict['G_GAN'] * not_warmup + loss_dict['G_GAN_Class'] * not_warmup + loss_dict.get('G_GAN_Feat',0) * not_warmup + \
            #          loss_dict.get('G_Cycle',0) * (min(not_warmup + texture_train,1)) + loss_dict.get('G_ID',0) + loss_dict.get('MinFlow',0) + \
            #          loss_dict.get('FlowTV',0) * not_warmup + loss_dict.get('Content_reconst',0) * not_warmup + \
            #          loss_dict.get('Age_reconst',0) * not_warmup + loss_dict.get('Age_kld',0) * not_warmup + \
            #          loss_dict.get('Landmarks_Loss',0) * not_warmup
            #
            # ############### Backward Pass ####################
            # # update generator weights
            # if not isinstance(loss_G, int):
            #     model.optimizer_G.zero_grad()
            #     loss_G.backward()
            #     model.optimizer_G.step()
            #
            # # update discriminator weights
            # model.optimizer_D.zero_grad()
            # loss_D.backward()
            # model.optimizer_D.step()

            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.item() if not (isinstance(v, float) or isinstance(v, int)) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch+1, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
                # visualizer.plot_current_errors(total_steps, errors)

            ### display output images
            if save_fake and opt.display_id > 0:
                if use_tensorboard:
                    class_a_suffix = ''
                    class_b_suffix = ''
                    classes = [data['A_class'][0],data['B_class'][0]]
                else:
                    class_a_suffix = ' class {}'.format(data['A_class'][0])
                    class_b_suffix = ' class {}'.format(data['B_class'][0])
                    classes = None

                visuals = OrderedDict()

                if opt.original_munit and data['A_class'][0] == 1:
                    visuals_A = OrderedDict([('real image' + class_a_suffix, util.tensor2im(gen_in.data[bSize,:3]))])
                    visuals_B = OrderedDict([('real image' + class_b_suffix, util.tensor2im(gen_in.data[0,:3]))])
                else:
                    visuals_A = OrderedDict([('real image' + class_a_suffix, util.tensor2im(gen_in.data[0,:3]))])
                    visuals_B = OrderedDict([('real image' + class_b_suffix, util.tensor2im(gen_in.data[bSize,:3]))])

                if opt.gan_mode == 'flow_only':
                    A_out_vis = OrderedDict([('warped image' + class_b_suffix, util.tensor2im(gen_flow.data[0])),
                                             ('unwarped image' + class_a_suffix, util.tensor2im(rec_flow.data[0]))])
                    B_out_vis = OrderedDict([('warped image' + class_a_suffix, util.tensor2im(gen_flow.data[bSize])),
                                             ('unwarped image' + class_b_suffix, util.tensor2im(rec_flow.data[bSize]))])
                    if opt.use_parsings:
                        A_out_parsings = OrderedDict([('real parsing' + class_a_suffix, util.tensor2im(data['A'][0,-3:,:,:])),
                                                      ('warped parsing' + class_b_suffix, util.tensor2im(gen_parsing.data[0])),
                                                      ('unwarped parsing' + class_a_suffix, util.tensor2im(rec_parsing.data[0]))])
                        B_out_parsings = OrderedDict([('real parsing' + class_b_suffix, util.tensor2im(data['B'][0,-3:,:,:])),
                                                      ('warped parsing' + class_a_suffix, util.tensor2im(gen_parsing.data[bSize])),
                                                      ('unwarped parsing' + class_b_suffix, util.tensor2im(rec_parsing.data[bSize]))])
                        A_out_vis.update(A_out_parsings)
                        B_out_vis.update(B_out_parsings)

                    if opt.use_landmarks and (not opt.json_landmarks):
                        A_out_landmarks = OrderedDict([('real landmarks' + class_a_suffix, util.tensor2im(data['landmarks_A'])),
                                                      ('warped landmarks' + class_b_suffix, util.tensor2im(gen_landmarks.data[0])),
                                                      ('unwarped landmarks' + class_a_suffix, util.tensor2im(rec_landmarks.data[0]))])
                        B_out_landmarks = OrderedDict([('real landmarks' + class_b_suffix, util.tensor2im(data['landmarks_B'][0,-3:,:,:])),
                                                      ('warped landmarks' + class_a_suffix, util.tensor2im(gen_landmarks.data[bSize])),
                                                      ('unwarped landmarks' + class_b_suffix, util.tensor2im(rec_landmarks.data[bSize]))])
                        A_out_vis.update(A_out_landmarks)
                        B_out_vis.update(B_out_landmarks)

                    A_out_flow = OrderedDict([('optical flow' + class_b_suffix, util.flow2im(opt_flow.data[0])),
                                              ('inverse optical flow' + class_a_suffix, util.flow2im(inv_opt_flow.data[0]))])
                    B_out_flow = OrderedDict([('optical flow to' + class_a_suffix, util.flow2im(opt_flow.data[bSize])),
                                              ('inverse optical flow' + class_b_suffix, util.flow2im(inv_opt_flow.data[bSize]))])
                    A_out_vis.update(A_out_flow)
                    B_out_vis.update(B_out_flow)

                    visuals_A.update(A_out_vis)
                    visuals_B.update(B_out_vis)
                    visuals.update(visuals_A)
                    visuals.update(visuals_B)
                elif (opt.gan_mode == 'texture_only' and not opt.use_flow_layers) or opt.gan_mode == 'seg_only':
                    if opt.gan_mode == 'seg_only':
                        A_out_parsings = OrderedDict([('real parsing' + class_a_suffix, util.tensor2im(data['A'][0,-3:,:,:]))])
                        B_out_parsings = OrderedDict([('real parsing' + class_b_suffix, util.tensor2im(data['B'][0,-3:,:,:]))])
                        visuals_A.update(A_out_parsings)
                        visuals_B.update(B_out_parsings)

                    A_out_vis = OrderedDict([('synthesized image' + class_b_suffix, conversion_func(gen_tex.data[0]))])
                    B_out_vis = OrderedDict([('synthesized image' + class_a_suffix, conversion_func(gen_tex.data[bSize]))])
                    if opt.lambda_id_tex > 0:
                        A_out_vis.update([('reconstructed image' + class_a_suffix, conversion_func(rec_tex.data[0]))])
                        B_out_vis.update([('reconstructed image' + class_b_suffix, conversion_func(rec_tex.data[bSize]))])
                    if opt.lambda_cyc_tex > 0:
                        A_out_vis.update([('cycled image' + class_a_suffix, conversion_func(cyc_tex.data[0]))])
                        B_out_vis.update([('cycled image' + class_b_suffix, conversion_func(cyc_tex.data[bSize]))])

                    visuals_A.update(A_out_vis)
                    visuals_B.update(B_out_vis)
                    visuals.update(visuals_A)
                    visuals.update(visuals_B)
                elif opt.gan_mode == 'seg_and_texture' or (opt.gan_mode == 'texture_only' and opt.use_flow_layers):
                    A_out_vis = OrderedDict([('synthesized image' + class_b_suffix, conversion_func(gen_tex.data[0]))])
                    B_out_vis = OrderedDict([('synthesized image' + class_a_suffix, conversion_func(gen_tex.data[bSize]))])
                    if opt.lambda_id_tex > 0:
                        A_out_vis.update([('reconstructed image' + class_a_suffix, conversion_func(rec_tex.data[0]))])
                        B_out_vis.update([('reconstructed image' + class_b_suffix, conversion_func(rec_tex.data[bSize]))])
                    if opt.lambda_cyc_tex > 0:
                        A_out_vis.update([('cycled image' + class_a_suffix, conversion_func(cyc_tex.data[0]))])
                        B_out_vis.update([('cycled image' + class_b_suffix, conversion_func(cyc_tex.data[bSize]))])

                    visuals_A.update(A_out_vis)
                    visuals_B.update(B_out_vis)
                    A_out_parsings = OrderedDict([('real parsing class' + class_a_suffix, util.tensor2im(data['A'][0,-3:,:,:]))])
                    B_out_parsings = OrderedDict([('real parsing class' + class_b_suffix, util.tensor2im(data['B'][0,-3:,:,:]))])
                    A_out_seg = OrderedDict([('synthesized parsing' + class_b_suffix, conversion_func(gen_seg.data[0])),
                                             ('reconstructed parsing' + class_a_suffix, conversion_func(rec_seg.data[0])),
                                             ('cycled parsing' + class_a_suffix, conversion_func(cyc_seg.data[0]))])
                    B_out_seg = OrderedDict([('synthesized parsing' + class_a_suffix, conversion_func(gen_seg.data[bSize])),
                                             ('reconstructed parsing' + class_b_suffix, conversion_func(rec_seg.data[bSize])),
                                             ('cycled parsing' + class_b_suffix, conversion_func(cyc_seg.data[bSize]))])
                    visuals_A.update(A_out_parsings)
                    visuals_B.update(B_out_parsings)
                    visuals_A.update(A_out_seg)
                    visuals_B.update(B_out_seg)
                    visuals.update(visuals_A)
                    visuals.update(visuals_B)
                else: #opt.gan_mode == 'flow_and_texture'
                    if not ('rec_flow' in vars()) or rec_flow is None:
                        A_out_vis = OrderedDict([('warped image' + class_b_suffix, util.tensor2im(gen_flow.data[0])),
                                                 ('synthesized image' + class_b_suffix, util.tensor2im(gen_tex.data[0])),
                                                 ('reconstructed image' + class_a_suffix, util.tensor2im(rec_tex.data[0])),
                                                 ('cycled image' + class_a_suffix, util.tensor2im(cyc_tex.data[0]))])
                        B_out_vis = OrderedDict([('warped image' + class_a_suffix, util.tensor2im(gen_flow.data[bSize])),
                                                 ('synthesized image' + class_a_suffix, util.tensor2im(gen_tex.data[bSize])),
                                                 ('reconstructed image' + class_b_suffix, util.tensor2im(rec_tex.data[bSize])),
                                                 ('cycled image' + class_b_suffix, util.tensor2im(cyc_tex.data[bSize]))])
                    else:
                        A_out_vis = OrderedDict([('warped image' + class_b_suffix, util.tensor2im(gen_flow.data[0])),
                                                 ('synthesized image' + class_b_suffix, util.tensor2im(gen_tex.data[0])),
                                                 ('unwarped image' + class_a_suffix, util.tensor2im(rec_flow.data[0])),
                                                 ('reconstructed image' + class_a_suffix, util.tensor2im(rec_tex.data[0]))])
                        B_out_vis = OrderedDict([('warped image' + class_a_suffix, util.tensor2im(gen_flow.data[bSize])),
                                                 ('synthesized image' + class_a_suffix, util.tensor2im(gen_tex.data[bSize])),
                                                 ('unwarped image' + class_b_suffix, util.tensor2im(rec_flow.data[bSize])),
                                                 ('reconstructed image' + class_b_suffix, util.tensor2im(rec_tex.data[bSize]))])

                    if opt.use_parsings:
                        A_out_parsings = OrderedDict([('real parsing' + class_a_suffix, util.tensor2im(data['A'][0,-3:,:,:])),
                                                      ('warped parsing' + class_b_suffix, util.tensor2im(gen_parsing.data[0]))])
                        if 'rec_parsing' in vars() and rec_parsing is not None:
                            A_out_parsings.update(OrderedDict([('unwarped parsing' + class_a_suffix, util.tensor2im(rec_parsing.data[0]))]))

                        B_out_parsings = OrderedDict([('real parsing' + class_b_suffix, util.tensor2im(data['B'][0,-3:,:,:])),
                                                      ('warped parsing' + class_a_suffix, util.tensor2im(gen_parsing.data[bSize]))])
                        if 'rec_parsing' in vars() and rec_parsing is not None:
                            B_out_parsings.update(OrderedDict([('unwarped parsing' + class_b_suffix, util.tensor2im(rec_parsing.data[bSize]))]))

                        if opt.use_parsings_tex_out:
                            A_out_parsings_tex = OrderedDict([('post texture parsings' + class_b_suffix, util.tensor2im(gen_tex_parsings.data[0]))])
                            B_out_parsings_tex = OrderedDict([('post texture parsings' + class_a_suffix, util.tensor2im(gen_tex_parsings.data[bSize]))])
                            A_out_parsings.update(A_out_parsings_tex)
                            B_out_parsings.update(B_out_parsings_tex)

                        A_out_vis.update(A_out_parsings)
                        B_out_vis.update(B_out_parsings)

                    if opt.use_landmarks:
                        A_out_landmarks = OrderedDict([('real landmarks' + class_a_suffix, util.tensor2im(data['landmarks_A'])),
                                                      ('warped landmarks' + class_b_suffix, util.tensor2im(gen_landmarks.data[0])),
                                                      ('unwarped landmarks' + class_a_suffix, util.tensor2im(rec_landmarks.data[0]))])
                        B_out_landmarks = OrderedDict([('real landmarks' + class_b_suffix, util.tensor2im(data['landmarks_B'][0,-3:,:,:])),
                                                      ('warped landmarks' + class_a_suffix, util.tensor2im(gen_landmarks.data[bSize])),
                                                      ('unwarped landmarks' + class_b_suffix, util.tensor2im(rec_landmarks.data[bSize]))])
                        A_out_vis.update(A_out_landmarks)
                        B_out_vis.update(B_out_landmarks)

                    A_out_flow = OrderedDict([('optical flow' + class_b_suffix, util.flow2im(opt_flow.data[0]))])
                    if 'inv_opt_flow' in vars() and inv_opt_flow is not None:
                        A_out_flow.update(OrderedDict([('inverse optical flow' + class_a_suffix, util.flow2im(inv_opt_flow.data[0]))]))
                    B_out_flow = OrderedDict([('optical flow' + class_a_suffix, util.flow2im(opt_flow.data[bSize]))])
                    if 'inv_opt_flow' in vars() and inv_opt_flow is not None:
                        B_out_flow.update(OrderedDict([('inverse optical flow' + class_b_suffix, util.flow2im(inv_opt_flow.data[bSize]))]))

                    A_out_vis.update(A_out_flow)
                    B_out_vis.update(B_out_flow)

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

            ### save model in multiple iterations in the first non warmup epoch
            if opt.gan_mode == 'flow_only' and (epoch == 1) and total_steps % first_epoch_save_freq == first_epoch_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch+1, total_steps))
                model.save('e1_it{}'.format(i))
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch+1, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if opt.display_id == 0:
            visuals = model.inference(sample_data)
            visualizer.save_matrix_image(visuals, epoch+1)

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

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch+1 == opt.niter_fix_global):
            model.update_fixed_params()

        ### instead of only training the texture model, train both models after certain iterations
        if (opt.niter_fix_flow != 0) and (epoch+1 == opt.niter_fix_flow):
            model.update_flow_params()

        if (opt.niter_fix_seg != 0) and (epoch+1 == opt.niter_fix_seg):
            model.update_flow_params()

        if opt.decay_method == 'linear':
            ### linearly decay learning rate after certain iterations
            if (epoch+1) > opt.niter:
                model.update_learning_rate()
        else:
            ### multiply learning rate by opt.decay_gamma after certain iterations
            if (epoch+1) in opt.decay_epochs:
                model.update_learning_rate()

if __name__ == "__main__":
    opt = TrainOptions().parse()
    train(opt)
