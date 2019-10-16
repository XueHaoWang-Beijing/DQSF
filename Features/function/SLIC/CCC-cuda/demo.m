clc; clear all; close all;

% parameter setting

sigma_s = 100;sigma_r = 0.3;%RF Smoothing Parameter
path = '/usr/home/matlab_Program/data_set/nju2000/RGB/';
Files = dir([path '*g']);
dd = 300;
for i=1:length(Files)
    for s=1:1
        img = imread([path Files(i).name]);
        w = size(img,1); h = size(img,2);
        img = imresize(img,[dd,dd]);
        box = zeros(dd,dd);
        
        for j=1:1
            uu = double(imread(['/usr/home/matlab_Program/tmp_try/grad/GS/' Files(i).name(1:end-3) 'png']));
            uu1 = uu;
            uu = imresize(uu,[dd,dd]);
            uu = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
            
            temp0 = RGBSmooth(img,uu,50,0.5,600);
            temp0 = (temp0-min(min(temp0)))/(max(max(temp0))-min(min(temp0)));
            temp1 = RGBSmooth(img,uu,50,0.5,300);
            temp1 = (temp1-min(min(temp1)))/(max(max(temp1))-min(min(temp1)));
            temp2 = RGBSmooth(img,uu,50,0.5,400);
            temp2 = (temp2-min(min(temp2)))/(max(max(temp2))-min(min(temp2)));
            temp3 = RGBSmooth(img,uu,50,0.5,500);
            temp3 = (temp3-min(min(temp3)))/(max(max(temp3))-min(min(temp3)));
            
            temp = temp0+temp1+temp2+temp3;
            if(max(max(uu1))==0)
                temp = zeros(dd,dd);
            end
            box = box+temp;
        end
        box = (box-min(min(box)))/(max(max(box))-min(min(box)));
        box = imresize(box,[w,h]);
        figure;set(gcf,'outerposition',get(0,'screensize'));
        subplot(131);imshow(imresize(img,[w,h]));
        subplot(132);imshow(uint8(uu1));
        subplot(133);imshow(box);
        close all;
        
    end
end

