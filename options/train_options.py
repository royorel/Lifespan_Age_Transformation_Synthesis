### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # for displays
        self.parser.add_argument('--display_freq', type=int, default=40, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=40, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_display_freq', type=int, default=5000, help='save the current display of results every save_display_freq_iterations')
        self.parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training & optimizer
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--epochs', type=int, default=400, help='# of epochs to train')
        self.parser.add_argument('--decay_gamma', type=float, default=0.5, help='decay the learning rate by this value')
        self.parser.add_argument('--decay_epochs', type=str, default='50,100', help='epochs to perform step lr decay')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--decay_adain_affine_layers', type=bool, default=True, help='when true adain affine layer learning rate is decayed by 0.01')

        # for discriminators
        self.parser.add_argument('--n_layers_D', type=int, default=6, help='number of styled convolution layers in the discriminator')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')

        # loss weights
        self.parser.add_argument('--lambda_cyc', type=float, default=10.0, help='weight for cycle loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_id', type=float, default=1.0, help='weight for identity encoding consistency loss')
        self.parser.add_argument('--lambda_age', type=float, default=1.0, help='weight for age encoding consistency loss')

        self.isTrain = True
