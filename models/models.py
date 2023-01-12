import torch

def create_model(opt):
    from .LATS_model import LATS, InferenceModel
    if opt.isTrain:
        model = LATS()
    else:
        model = InferenceModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    return model
