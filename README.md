# Depth Quality-aware Selective Saliency Fusion for RGB-D Image Salient Object Detection

## Prerequisites
| [Caffe](https://github.com/BVLC/caffe) | [CUDA10](https://developer.nvidia.com/cuda-downloads) | [CUDNN7.5](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) | [Matlab2016b](https://www.mathworks.com/) |

## Usage
1. Clone this code by `git clone https://github.com/XueHaoWang-Beijing/DQSF.git --recursive`,  
assume your source code directory is `$DQSF/`
2. Calculate our proposed depth quality-aware features (or download them directly from here [Google drive](https://drive.google.com/file/d/1w6h6C9NkQhFqADo-PXFK31jy3y3xs1os/view?usp=sharing
https://drive.google.com/file/d/1TkEpsfspPOphzT_qitYWUPPscC9vZsXX/view?usp=sharing),[Baiduyun PW:zeht](https://pan.baidu.com/s/1Vc5u3gAHMI5iJhf8-Cm_ow) ).
  * For RQ and SM features, run the code `./DQSF/Features/RDQ.m`.
  * For SMM features, run the code `./DQSF/Features/SMM_Network/Test/tesDemo.m` and `./DQSF/Features/SMM_Network/Test/tesDemor.m` to generate the RGBD and RGBDrand predictions.  
  Then run the code `./DQSF/Features/SSMG.m` to calculate the SMM features.  
## Training
1. Download training data([Google drive](https://drive.google.com/file/d/1tmGjqfIAO2cTDZ8QmHXsUlBfZPTbtVeU/view?usp=sharing) or [Baiduyun PW:4w6j](https://pan.baidu.com/s/1vumhbAUJqCSslhPEIMyY4g)), and extract it to `./DQSF/Dataset/`
2. Download [initial model](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) and put it into `./DQSF/Network/Train/Model/`
3. Start to train with `sh ./DQSF/Network/Train/finetune.sh`.
## Testing
1. Download [pretrained model](https://drive.google.com/open?id=1VaPl5lUkPW_uMlZuZ6hN-pRzxgdTXd4I) and [RGBD datasets](https://drive.google.com/file/d/1tmGjqfIAO2cTDZ8QmHXsUlBfZPTbtVeU/view?usp=sharing)
2. Generate saliency maps by run the code `./DQSF/Network/Test/tesDemo.m`

