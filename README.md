# Depth Quality-aware Selective Saliency Fusion for RGB-D Image Salient Object Detection

This is a Caffe implementation of our TIP 2019 paper.
## Prerequisites
|[Caffe](https://github.com/BVLC/caffe)|[CUDA10](https://developer.nvidia.com/cuda-downloads)|[CUDNN7.5](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)|[Matlab2016b](https://www.mathworks.com/)|

## Usage
1. Clone this code by `git clone https://github.com/XueHaoWang-Beijing/DQSF.git --recursive`,  
assume your source code directory is `$DQSF/`
2. Calculate our proposed depth quality-aware features (or download them directly from here [Google drive](),[Baiduyun PW:zeht](https://pan.baidu.com/s/1Vc5u3gAHMI5iJhf8-Cm_ow) ).
  * For RQ and SM features, run the code `./DQSF/Features/RDQ.m`.
  * For SMM features, run the code `./DQSF/Features/SMM_Network/Test/tesDemo.m` and `./DQSF/Features/SMM_Network/Test/tesDemor.m` to generate the RGBD and RGBDrand predictions.  
  Then run the code `./DQSF/Features/SSMG.m` to calculate the SMM features.  
## Training
1. Download training data([Google drive]() or [Baiduyun PW:4w6j](https://pan.baidu.com/s/1vumhbAUJqCSslhPEIMyY4g)), and extract it to `./DQSF/Dataset/`
2. Download [initial model](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) and put it into `./DQSF/Network/Train/Model/`
3. Start to train with `sh ./DQSF/Network/Train/finetune.sh`.
## Testing
1. Download [pretrained model](https://drive.google.com/open?id=1VaPl5lUkPW_uMlZuZ6hN-pRzxgdTXd4I) `./DQSF/Network/Test/tesDemo.m`;
2. Generate saliency maps by run the code `./DQSF/Network/Test/tesDemo.m`;

