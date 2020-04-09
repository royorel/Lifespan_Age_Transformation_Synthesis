### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--random_seed', type=int, default=-1, help='random seed for generating different outputs from the same model.')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
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
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")
        self.parser.add_argument("-d", "--data_type", default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.isTrain = False
