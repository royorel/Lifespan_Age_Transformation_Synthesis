# Lifespan Age Transformation Synthesis
### [Project Page](https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/) | [Paper](https://arxiv.org/pdf/2003.09764.pdf) | [FFHQ-Aging Dataset](https://github.com/royorel/FFHQ-Aging-Dataset)

[![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/royorel/Lifespan_Age_Transformation_Synthesis/blob/master/LATS_demo.ipynb)<br>

[Roy Or-El](https://homes.cs.washington.edu/~royorel/)<sup>1</sup> ,
[Soumyadip Sengupta](https://homes.cs.washington.edu/~soumya91/)<sup>1</sup>,
[Ohad Fried](https://www.ohadf.com/)<sup>2</sup>,
[Eli Shechtman](https://research.adobe.com/person/eli-shechtman/)<sup>3</sup>,
[Ira Kemelmacher-Shlizerman](https://homes.cs.washington.edu/~kemelmi/)<sup>1</sup><br>
<sup>1</sup>University of Washington, <sup>2</sup>Stanford University, <sup>3</sup>Adobe Research

<div align="center">
<img src=./gifs/001_input.gif width="150">
<img src=./gifs/69005.gif width="150">
<img src=./gifs/69131.gif width="150">
<img src=./gifs/69508.gif width="150">
<img src=./gifs/69587.gif width="150"><br>
<img src=./gifs/010_input.gif width="150">
<img src=./gifs/69036.gif width="150">
<img src=./gifs/69338.gif width="150">
<img src=./gifs/69577.gif width="150">
<img src=./gifs/69859.gif width="150">
</div>

## Updates
 - August 19th, 2020: Added alternate download urls for pretrained models to bypass Google Drive "quota exceeded" error.

## Overview
Lifespan Age Transformation Synthesis is a GAN based method designed to simulate the continuous aging process from a single input image.<br>
This code is the official PyTorch implementation of the paper:
> **Lifespan Age Transformation Synthesis**<br>
> Roy Or-El, Soumyadip Sengupta, Ohad Fried, Eli Shechtman, Ira Kemelmacher-Shlizerman<br>
> ECCV 2020<br>
> https://arxiv.org/pdf/2003.09764.pdf

## Ethics & Bias statement
### Intended use:
 - This algorithm is designed to hallucinate the aging process and produce an **approximation** of a person's appearance throughout his/her/their lifespan.
 - The main use cases of this method are for art and entertainment purposes (CGI effects, Camera filters, etc.). This method might also be useful for more critical applications, e.g. approximating the appearance of missing people. However, we would like to stress that as a non perfect data-driven method, results might be inaccurate and biased. The output of our method should be critically analyzed by a trained professional, and not be treated as an absolute ground truth.
 - **_The results of this method should not be used as grounds for detention/arrest of a person or as any other form of legal evidence under any circumstances._**

### Algorithm & data bias:<br>
We have devoted considerable efforts in our algorithm design to preserve the identity of the person in the input image, and to minimize the influence of the inherent dataset biases on the results. These measures include:
1. Designing the identity encoder architecture to preserve the local structures of the input image.
2. Including training losses that were designed to maintain the person's identity.
    - Latent Identity loss: encourages identity features that are consistent across ages.
    - Cycle loss: drives the network to reproduce the original image from any aged output.
    - Self-reconstruction loss: makes the network learn to reconstruct the input when the target age class is the same as the source age class.
3. The FFHQ dataset contains gender imbalance within age classes. To prevent introducing these biases in the output, e.g. producing male facial features for females or vice versa, we have trained two separate models, one for males and one for females. The decision of which model to apply is left for the user. We acknowledge that this design choice restricts our algorithm from simulating the aging process of people whose gender is non-binary. Further work is required to make sure future algorithms will be able to simulate aging for the entire gender spectrum.

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
9. tqdm
10. dlib

If any of these packages are not installed on your computer, you can install them using the supplied `requirements.txt` file:<br>
```pip install -r requirements.txt```

## Quick Demo
You can try running the method on your own images!!!<br>

You can either run the demo localy or explore it in Colab [![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/royorel/Lifespan_Age_Transformation_Synthesis/blob/master/LATS_demo.ipynb)<br>

Running locally:
1. Download the pre-trained models. ```python download_models.py```
2. Create a txt file with the paths to all images you want to try (for example, see ```males_image_list.txt``` or ```females_image_list.txt```)
3. Run the model:
   - Open, ```./run_scripts/in_the_wild.sh``` (Linux) or ```./run_scripts/in_the_wild.bat``` (windows).
   - Select which model to use in the ```--name``` flag (```males_model``` or ```females_model```).
   - Enter the path to the txt file you created in step 2 after the ```--image_path_file``` flag.
   - Run the script.
4. The outputs can be seen in ```results/males_model/test_latest/traversal/``` or ```results/females_model/test_latest/traversal/``` (according to the selected model).

Please refer to [Using your own images](#using-your-own-images) for guidelines on what images are good to use.<br>

If you get a CUDA out of memory error, slightly increase the ```--interp_step``` parameter until it fits your GPU. This parameter controls the number of interpolated frames between every 2 anchor classes. Increasing it will reduce the length of the output video.

## Using your own images
For best results, use images according to the following guidelines:
1. The image should contain a single face.
2. Image was taken from a digital camera (phone cameras are fine). Old images from film cameras would produce low quality results.
3. Pure RGB images only. No black & white, grayscale, sepia, or filtered images (e.g. Instagram filters).
4. Person's head should directly face the camera. Looking sideways/downwards/upwards degrades the results.
5. The person's face should not be occluded (or partially occluded) by any item.
6. Both eyes should be open and visible. (Eyeglasses are ok, but might produce artifacts. No sunglasses)


## Training/Testing on FFHQ-Aging
1. Download the FFHQ-Aging dataset. Go to the [FFHQ-Aging dataset repo](https://github.com/royorel/FFHQ-Aging-Dataset) and follow the instructions to download the data.

2. Prune & organize the raw FFHQ-Aging dataset into age classes:
```
cd datasets
python create_dataset.py --folder <path to raw FFHQ-Aging directory> --labels_file <path to raw FFHQ-Aging labels csv file> [--train_split] [num of training images (default=69000)]
```

3. Download pretrained models (Optional)<br>
```python download_models.py```

### Training
1. Open a visdom port to view loss plots and intermediate results. Run ```visdom``` and monitor results at [http://localhost:8097](http://localhost:8097). If you run the code on a remote server open ```http://hostname:8097``` instead.
2. Open ```run_scripts/train.sh``` (Linux) or ```run_scripts/train.bat``` (windows) and set:
   - The GPUs you intend to use ```--gpu_ids``` as well as the ```CUDA_VISIBLE_DEVICES``` environment variable.<br>
     NOTE: the scripts are currently set to use 4 GPUs
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - The batch size ```--batchSize``` according to your GPU's maximum RAM capacity and the number of GPU's available.
   - If you wish to continue training from an existing checkpoint, add the ```--continue_training``` flag and specify the checkpoint you wish to continue training from in the ```--which_epoch``` flag, e.g: ```--which_epoch 100``` or ```--which_epoch latest```.
3. Train the model: Run```./run_scripts/train.sh``` (Linux) or ```./run_scripts/train.bat``` (windows)

### Testing
1. Open ```run_scripts/test.sh``` (Linux) or ```run_scripts/test.bat``` (windows) and set:
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - Which checkpoint to load the model from ```--which_epoch```. This can be either an epoch number e.g. ```400``` or the latest saved model ```latest```.
2. Test the model: Run```./run_scripts/test.sh``` (Linux) or ```./run_scripts/test.bat``` (windows)
3. The outputs can be seen in ```results/<model name>/test_<model_checkpoint>/index.html```

### Generate Video
1. Prepare a ```.txt``` file with a list of image paths to generate videos for. See examples in ```males_image_list.txt``` and ```females_image_list.txt```
2. Open ```run_scripts/traversal.sh``` (Linux) or ```run_scripts/traversal.bat``` (windows) and set:
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - Which checkpoint to load the model from ```--which_epoch```. This can be either an epoch number e.g. ```400``` or the latest saved model ```latest```.
   - The relative path to the image list ```--image_path_file```
3. Run ```./run_scripts/traversal.sh``` (Linux) or ```./run_scripts/traversal.bat``` (windows)
4. The output videos will be saved to ```results/<model name>/test_<model_checkpoint>/traversal/```

### Generate Full Progression
This will generate an image of progressions to all anchor classes
1. Prepare a ```.txt``` file with a list of image paths to generate videos for. See examples in ```males_image_list.txt``` and ```females_image_list.txt```
2. Open ```run_scripts/deploy.sh``` (Linux) or ```run_scripts/deploy.bat``` (windows) and set:
   - The dataset relative path ```--dataroot```
   - The model name ```--name```
   - Which checkpoint to load the model from ```--which_epoch```. This can be either an epoch number e.g. ```400``` or the latest saved model ```latest```.
   - The relative path to the image list ```--image_path_file```
3. Run ```./run_scripts/deploy.sh``` (Linux) or ```./run_scripts/deploy.bat``` (windows)
4. The output images will be saved to ```results/<model name>/test_<model_checkpoint>/deploy/```

## Training/Testing on New Datasets
If you wish to train the model on a new dataset, arange it in the following structure:
```                                                                                           
├── dataset_name                                                                                                                                                                                                       
│   ├── train<class1> 
|   |   └── image1.png
|   |   └── image2.png
|   |   └── ...                                                                                                
│   │   ├── parsings
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ...                                                                                                                             
...
│   ├── train<classN> 
|   |   └── image1.png
|   |   └── image2.png
|   |   └── ...                                                                                                
│   │   ├── parsings
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ... 
│   ├── test<class1> 
|   |   └── image1.png
|   |   └── image2.png
|   |   └── ...                                                                                                
│   │   ├── parsings
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ...                                                                                                                             
...
│   ├── test<classN> 
|   |   └── image1.png
|   |   └── image2.png
|   |   └── ...                                                                                                
│   │   ├── parsings
│   │   │   └── image1.png
│   │   │   └── image2.png
│   │   │   └── ... 
``` 

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{orel2020lifespan,
  title={Lifespan Age Transformation Synthesis},
  author={Or-El, Roy
          and Sengupta, Soumyadip
          and Fried, Ohad
          and Shechtman, Eli
          and Kemelmacher-Shlizerman, Ira},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## Acknowledgments
This code is inspired by [pix2pix-HD](https://github.com/NVIDIA/pix2pixHD) and [style-based-gan-pytorch](https://github.com/rosinality/style-based-gan-pytorch).
