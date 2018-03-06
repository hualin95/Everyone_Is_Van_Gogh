![](https://github.com/hualin95/Everyone_Is_Van_Gogh/blob/master/docs/logo.png)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/hualin95/Everyone_Is_Van_Gogh/blob/master/LICENSE) 
![Build Status](https://img.shields.io/appveyor/ci/gruntjs/grunt/master.svg)
# Everyone_Is_Van_Gogh
Everyone_Is_Van_Gogh is a tensorflow implementation of style transfer which described in the next paper:
* [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf)

And I use VGG19 which was proposed in this paper:
* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

To use the VGG19 networks, you have to download the npy files for VGG19 NPY from [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).

# Environment
* 1.tensorflow v1.4
* 2.Cudnn v6.0
* 3.Python v3.6 
* 4.Python packages:numpy,PIL,scipy
* 4.Bash(git、ssh)

# Getting Started
* 1.Clone this project : `git clone git@github.com:hualin95/Everyone_Is_Van_Gogh.git`
* 2.Enter the folder Everyone_Is_Van_Gogh : `cd Everyone_Is_Van_Gogh`
* 3.Run the file train.py : 
```
python train.py --VGG19_Model_Path [your filepath of vgg19.npy]  --content_image [your filepath of content_image] --style_image [your filepath of style_image]
```
*For example (I have move the file vgg19.npy to the folder Everyone_Is_Van_Gogh)*: 
```
python train.py --VGG19_Model_Path "vgg19.npy"  --content_image "images/content/johannesburg.jpg" --style_image "images/style/starry-night.jpg"
```
*More : I guess you want to train the project with GPU if you have one, so you can run the file train.py like this*:
```
CUDA_VISIBLE_DEVICES=[GPU_num] python train.py --VGG19_Model_Path [your filepath of vgg19.npy]  --content_image [your filepath of content_image] --style_image [your filepath of style_image]
```
*For example* : 
```
CUDA_VISIBLE_DEVICES=4 python train.py --VGG19_Model_Path "vgg19.npy"  --content_image "images/content/johannesburg.jpg" --style_image "images/style/starry-night.jpg"
```
* 4.The generated image is saved in Everyone_Is_Van_Gogh/images/generated/

# Sample
## Style_iamge
![](https://github.com/hualin95/Everyone_Is_Van_Gogh/blob/master/images/style/starry-night.jpg)

[The Starry Night](https://www.moma.org/learn/moma_learning/vincent-van-gogh-the-starry-night-1889) is an oil on canvas by the Dutch post-impressionist painter Vincent van Gogh in June 1889.
## Content_image
![](https://github.com/hualin95/Everyone_Is_Van_Gogh/blob/master/images/content/ucas1.jpg) 
This is picture of University of Chinese Academy of Sciences where I am studying now and finish this project.
## Generated_image(after 1000 iteration)
![](https://github.com/hualin95/Everyone_Is_Van_Gogh/blob/master/images/generated/ucas1_starry-night.png)
