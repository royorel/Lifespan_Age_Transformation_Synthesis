# Lifespan Age Transformation Synthesis
### [Project Page](https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/) | [Paper](https://arxiv.org/pdf/2003.09764.pdf) | [Data](https://github.com/royorel/FFHQ-Aging-Dataset)

[Roy Or-El](https://homes.cs.washington.edu/~royorel/)<sup>1</sup> ,
[Soumyadip Sengupta](https://homes.cs.washington.edu/~soumya91/)<sup>1</sup>,
[Ohad Fried](https://www.ohadf.com/)<sup>2</sup>,
[Eli Shechtman](https://research.adobe.com/person/eli-shechtman/)<sup>3</sup>,
[Ira Kemelmacher-Shlizerman](https://homes.cs.washington.edu/~kemelmi/)<sup>1</sup><br>
<sup>1</sup>University of Washington, <sup>2</sup>Stanford University, <sup>3</sup>Adobe Research

## Overview
Lifespan Age Transformation Synthesis is a GAN based method designed to simulate the countinuous aging process from a single input image.<br>
This code is the official PyTorch implementation of the paper:
> **Lifespan Age Transformation Synthesis**<br>
> Roy Or-El, Soumyadip Sengupta, Ohad Fried, Eli Shechtman, Ira Kemelmacher-Shlizerman<br>
> https://arxiv.org/pdf/2003.09764.pdf

## Ethics & Bias statement
### Intended use:
 - This algorithm is designed to hallucinate the aging proccess and produce an **_approximation_** of a person's appearance throughout his/her/their lifespan.
 - The main usecases of this method are for art and entertainment purposes (CGI effects, Camera filters, etc.). This method might also be useful for more critical applications, e.g. approximating the appearance of missing people. However, we would like to stress that as a non perfect data-driven method, results might be inaccurate and biased. The output of the our method should be critically analyzed by a trained professional, and not be treated as an absolute ground truth.
 - **_The results of this method should not be used as grounds for detention/arrest of a person or as any other from of legal evidence under any circumstances_**

### Algorithm & data bias:<br>
We have devoted considerable efforts in our algorithm design to preserve the identity of the person in the input image, and to minimize the influence of the inherent dataset biases on the results. These measures include:
1. Designing the identity encoder architecture to preserve the local structures of the input image.
2. Including training losses that were designed to maintain the person's identity. These losses make sure that the encoded identity features are consistent across ages (latent identity loss), that the network can reproduce the original image from its output (cycle loss) and that the network learns to reconstruct the input if the target age class is the same as the source age class (self-reconstruction loss).
3. The FFHQ dataset contains gender imbalance within age classes. To prevent introducing these biases in the output, e.g. producing male facial features for females or vice versa, we have trained two separate models, one for males and one for females. The decision of which model to apply is left for the user. We acknowledge that gender is non-binary and that this design choice restrict our algorithm from simulating the aging process of people whose gender is non-binary. Further work is required to make sure future algorithms will be able to simulate aging for the entire gender spectrum.

Despite these measures, the network might still introduce other biases that we did not consider when designing the algorithm. If you spot any bias in the results, please reach out to help future research!

## Pre-Requisits
You must have a **GPU with CUDA support** in order to run the code.

This code requires **PyTorch** and **torchvision** to be installed, please go to [PyTorch.org](https://pytorch.org/) for installation info.<br>
We tested our code on PyTorch 1.4.0 and torchvision 0.5.0, but the code should run on any PyTorch version above 1.0.0, and any torchvision version above 0.4.0.

The following python packages should also be installed:
1. opencv-python
2. visdom
3. dominate
4. numpy
5. scipy
6. pillow
7. unidecode
8. requests

If any of these packages are not installed on your computer, you can install them using the supplied `requirements.txt` file:<br>
```pip install -r requirements.txt```

## Quick Demo (Coming soon)
Try running the method on your own image...<br>
```Coming soon```

## Get Started
1. To start working with the code you need to download the FFHQ-Aging dataset. Go to the [FFHQ-Aging dataset repo](https://github.com/royorel/FFHQ-Aging-Dataset) and follow the instructions to download the data.

2. Prune & organize the raw FFHQ-Aging dataset into age classes:
```
cd datasets
python create_dataset.py --folder <path to raw FFHQ-Aging directory> --labels_file <path to raw FFHQ-Aging labels csv file> [--train_split] [num of training images (default=69000)]
```

3. Download pretrained models (Optional, coming soon)<br>
```coming soon```

## Usage
### Training:
1. Open a visdom port to view loss plots and intermediate results. Run ```visdom``` and monitor results at [http://localhost:8097](http://localhost:8097). If you run the code on a remote server open ```http://hostname:8097``` instead.
2. Open ```run_scripts/train.sh``` (Linux) or ```run_scripts/train.bat``` (windows) and set:
   - The GPUs you intend to use ```--gpu_ids``` as well as the ```CUDA_VISIBLE_DEVICES``` environment variable.
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - The batch size ```--batchSize``` according to your GPU's maximum RAM capacity and the number of GPU's available.
3. Train the model: Run```./run_scripts/train.sh``` (Linux) or ```./run_scripts/train.bat``` (windows)

### Testing:
1. Open ```run_scripts/test.sh``` (Linux) or ```run_scripts/test.bat``` (windows) and set:
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - Which checkpoint to load the model from ```--which_epoch```. This can be either an epoch number e.g. '''400''' or the latest saved model '''latest'''.
2. Test the model: Run```./run_scripts/train.sh``` (Linux) or ```./run_scripts/train.bat``` (windows)
3. The outputs can be seen in results/<model name>/test_<model_checkpoint>/index.html

### Generate Video
1. Prepare a ```.txt``` file with a list of image paths to generate videos for, omit the file extentions. See examples in ```males_image_list.txt``` and ```females_image_list.txt```
2. Open ```run_scripts/traversal.sh``` (Linux) or ```run_scripts/traversal.bat``` (windows) and set:
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - Which checkpoint to load the model from ```--which_epoch```. This can be either an epoch number e.g. '''400''' or the latest saved model '''latest'''.
   - The relative path to the image list ```--image_path_file```
3. Run ```./run_scripts/traversal.sh``` (Linux) or ```./run_scripts/traversal.bat``` (windows)
4. The output videos will be saved to results/<model name>/traversal/

### Generate anchor age classes images
1. Prepare a ```.txt``` file with a list of image paths to generate videos for, omit the file extentions. See examples in ```males_image_list.txt``` and ```females_image_list.txt```
2. Open ```run_scripts/deploy.sh``` (Linux) or ```run_scripts/deploy.bat``` (windows) and set:
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - Which checkpoint to load the model from ```--which_epoch```. This can be either an epoch number e.g. '''400''' or the latest saved model '''latest'''.
   - The relative path to the image list ```--image_path_file```
3. Run ```./run_scripts/deploy.sh``` (Linux) or ```./run_scripts/deploy.bat``` (windows)
4. The output images will be saved to results/<model name>/deploy/

## Citation
```
@inproceedings{orel2020lifespan,
  title={Lifespan Age Transformation Synthesis},
  author={Or-El, Roy
          and Sengupta, Soumyadip
          and Fried, Ohad
          and Shechtman, Eli
          and Kemelmacher-Shlizerman, Ira},
  eprint={2003.09764},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
  year={2020}
}
```
