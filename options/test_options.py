### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--random_seed', type=int, default=-1, help='random seed for generating different outputs from the same model.')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--in_the_wild', action='store_true', help='for evaluating on in the wild images')
        self.parser.add_argument('--traverse', action='store_true', help='when true, run latent space traversal on a list of images')
        self.parser.add_argument('--full_progression', action='store_true', help='when true, deploy mode saves all outputs as a single image')
        self.parser.add_argument('--make_video', action='store_true', help='when true, make a video from the traversal results')
        self.parser.add_argument('--compare_to_trained_outputs', action='store_true', help='when true, interpolate a trained class in order to compare to trained outputs')
        self.parser.add_argument('--compare_to_trained_class', type=int, default=1, help='what class to compare to')
        self.parser.add_argument('--trained_class_jump', type=int, default=1, choices=[1,2],help='how many classes to jump')
        self.parser.add_argument('--interp_step', type=float, default=0.5, help='step size of interpolated w space vectors between each 2 true w space vectors')
        self.parser.add_argument('--deploy', action='store_true', help='when true, run forward pass on a list of images')
        self.parser.add_argument('--image_path_file', type=str, help='a file with a list of images to perform run through the network and/or latent space traversal on')
        self.parser.add_argument('--debug_mode', action='store_true', help='when true, all intermediate outputs are saved to the html file')
        self.isTrain = False
