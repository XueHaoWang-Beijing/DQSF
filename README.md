# Depth Quality-aware Selective Saliency Fusion for RGB-D Image Salient Object Detection

This is a Caffe implementation of our TIP 2019 paper.
## Prerequisites
1. [Caffe](https://github.com/BVLC/caffe)  
2. [CUDA10](https://developer.nvidia.com/cuda-downloads)  
3. [CUDNN7.5](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)  
4. [Matlab2016b](https://www.mathworks.com/)
## Usage
1. Clone this code by `git clone https://github.com/XueHaoWang-Beijing/DQSF.git --recursive`, assume your source code directory is `$DQSF`
2. Calculate our proposed depth quality-aware features.
  * For RQ and SM features, run the code `./DQSF/Features/RDQ.m`.
  * For SMM features, run the code `./DQSF/Features/SMM_Network/Test/tesDemo.m` and `./DQSF/Features/SMM_Network/Test/tesDemor.m` to generate the RGBD and RGBDrand predictions.
  Then run the code `./DQSF/Features/SSMG.m` to calculate the SMM features.  
Also, we provide our proposed depth quality-aware features, you can download [them](abc) directly instead of calculating them by yourself.
