clc; clear all;

dir_rgbd = '/home/wang/Documents/RGBDset/1Train_2times/RGBD/';
dir_rgbdr = '/home/wang/Documents/RGBDset/1Train_2times/RGBDr/';

list = dir([dir_rgbd '*.png']);

for i=1:length(list)
    rgbd = imread([dir_rgbd list(i).name]);
    rgbdr = imread([dir_rgbdr list(i).name]);
    
    ssm = abs(im2double(rgbd)-im2double(rgbdr));
    ssm = (ssm-min(ssm(:)))/(max(ssm(:))-min(ssm(:)));
    imwrite(uint8(255*ssm),['./SMM/' list(i).name]);
end