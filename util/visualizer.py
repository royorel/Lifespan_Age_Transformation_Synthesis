### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import os
import cv2
import time
import unidecode
from . import util
from . import html
from pdb import set_trace as st

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.numClasses = opt.numClasses
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.save_freq = opt.save_display_freq

        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        if self.isTrain:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, it, classes, ncols):
        if self.display_single_pane_ncols > 0:
            h, w = next(iter(visuals.values())).shape[:2]
            table_css = """<style>
table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
            # ncols = self.display_single_pane_ncols
            title = self.name
            label_html = ''
            label_html_row = ''
            nrows = int(np.ceil(len(visuals.items()) / ncols))
            images = []
            idx = 0
            for label, image_numpy in visuals.items():
                label_html_row += '<td>%s</td>' % label
                if image_numpy.ndim < 3:
                    image_numpy = np.expand_dims(image_numpy, 2)
                    image_numpy = np.tile(image_numpy, (1, 1, 3))

                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
                if idx % ncols == 0:
                    label_html += '<tr>%s</tr>' % label_html_row
                    label_html_row = ''
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
            while idx % ncols != 0:
                images.append(white_image)
                label_html_row += '<td></td>'
                idx += 1
            if label_html_row != '':
                label_html += '<tr>%s</tr>' % label_html_row

            self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                            padding=2, opts=dict(title=title + ' images'))
            label_html = '<table>%s</table>' % label_html
            self.vis.text(table_css + label_html, win = self.display_id + 2,
                          opts=dict(title=title + ' labels'))
        else:
            idx = 1
            for label, image_numpy in visuals.items():
                self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                idx += 1


    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_matrix_image(self, visuals, epoch):
        for i in range(len(visuals)):
            visual = visuals[i]
            orig_img = visual['orig_img_cls_' + str(i)]
            curr_row_img = orig_img
            for cls in range(self.numClasses):
                next_im = visual['tex_trans_to_class_' + str(cls)]
                curr_row_img = np.concatenate((curr_row_img, next_im), 1)

            if i == 0:
                matrix_img = curr_row_img
            else:
                matrix_img = np.concatenate((matrix_img, curr_row_img), 0)

        if epoch != 'latest':
            epoch_txt = 'epoch_' + str(epoch)
        else:
            epoch_txt = epochs

        image_path = os.path.join(self.img_dir,'sample_batch_{}.png'.format(epoch_txt))
        util.save_image(matrix_img, image_path)

    def save_row_image(self, visuals, image_path, traverse=False):
        visual = visuals[0]
        orig_img = visual['orig_img']
        h, w, c = orig_img.shape
        traversal_img = np.concatenate((orig_img, np.full((h, 10, c), 255, dtype=np.uint8)), 1)
        if traverse:
            out_classes = len(visual) - 1
        else:
            out_classes = self.numClasses
        for cls in range(out_classes):
            next_im = visual['tex_trans_to_class_' + str(cls)]
            traversal_img = np.concatenate((traversal_img, next_im), 1)

        util.save_image(traversal_img, image_path)

    def make_video(self, visuals, video_path):
        fps = 20#25
        visual = visuals[0]
        orig_img = visual['orig_img']
        h, w = orig_img.shape[0], orig_img.shape[1]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        out_classes = len(visual) - 1
        for cls in range(out_classes):
            next_im = visual['tex_trans_to_class_' + str(cls)]
            writer.write(next_im[:,:,::-1])

        writer.release()

    # save image to the disk
    def save_images_deploy(self, visuals, image_path):
        for i in range(len(visuals)):
            visual = visuals[i]
            for label, image_numpy in visual.items():
                save_path = '%s_%s.png' % (image_path, label)
                util.save_image(image_numpy, save_path)


    # save image to the disk
    def save_images(self, webpage, visuals, image_path, gt_visuals=None, gt_path=None):
        cols = self.numClasses+1
        image_dir = webpage.get_image_dir()
        if gt_visuals == None or gt_path == None:
            for i in range(len(visuals)):
                visual = visuals[i]
                short_path = os.path.basename(image_path[i])
                name = unidecode.unidecode(os.path.splitext(short_path)[0]) #removes accents which cause html load error
                webpage.add_header(name)
                ims = []
                txts = []
                links = []
                for label, image_numpy in visual.items():
                    image_name = '%s_%s.png' % (name, label)
                    save_path = os.path.join(image_dir, image_name)
                    util.save_image(image_numpy, save_path)

                    ims.append(image_name)
                    txts.append(label)
                    links.append(image_name)

                webpage.add_images(ims, txts, links, width=self.win_size,cols=cols)
        else:
            batchSize = len(image_path)

            # save ground truth images
            if gt_path is not None:
                gt_short_path = os.path.basename(gt_path[0])
                gt_name = os.path.splitext(gt_path)[0]
                gt_ims = []
                gt_txts = []
                gt_links = []
                for label, image_numpy in gt_visuals.items():
                    image_name = '%s_%s.png' % (gt_name, label)
                    save_path = os.path.join(image_dir, image_name)
                    util.save_image(image_numpy, save_path)

                    gt_ims.append(image_name)
                    gt_txts.append(label)
                    gt_links.append(image_name)

            for i in range(batchSize):
                short_path = os.path.basename(image_path[i])
                name = os.path.splitext(short_path)[0]

                # webpage.add_header(name)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals[i].items():
                    image_name = '%s_%s.png' % (name, label)
                    save_path = os.path.join(image_dir, image_name)
                    util.save_image(image_numpy, save_path)

                    ims.append(image_name)
                    txts.append(label)
                    links.append(image_name)
                print("saving results for: " + name)

                if gt_path is not None:
                    webpage.add_header(gt_name)
                    webpage.add_images(gt_ims, gt_txts, gt_links, width=self.win_size, cols=batchSize)

                webpage.add_header(name)
                webpage.add_images(ims, txts, links, width=self.win_size, cols=self.numClasses + 1)
