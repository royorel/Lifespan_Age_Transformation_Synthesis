### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

def create_model(opt):
    if opt.model == 'flowgan_hd':
        from .FlowGAN_HD_model import FlowGANHDModel, InferenceModel
        if opt.isTrain:
            model = FlowGANHDModel()
        else:
            model = InferenceModel()
    else:
        from .ui_model import UIModel
        model = UIModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    # if opt.isTrain and len(opt.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
