clc;clear all;close all;
addpath('./function/RGBSmooth/');
addpath('./function/Entropy/');
addpath('./function/SLIC/CCC-cuda/');

dir_hed = '../../RGBDset/1Train_2times/HED/';
dir_rgb = '../../RGBDset/1Train_2times/RGB/';
dir_depth = '../../RGBDset/1Train_2times/depth/';

dir_RQ = './RQ/';
dir_SM = './SM/';
% dir_SMM = dir_SMM_list{dataset_id};
mkdir(dir_RQ);
mkdir(dir_SM);
% mkdir(dir_SMM);
mkdir('./t123/');

list = dir([dir_rgb '*.png']);
parfor i = 1:length(list)
    %% RQ
    str = list(i).name;
    disp(str);
    rgb = imread([dir_rgb list(i).name]);
    depth = imread([dir_depth list(i).name]);
    hed = imread([dir_hed list(i).name]);

    [xd,yd] = gradient(double(depth));
%     depth_grad = abs(xd)+abs(yd);
    depth_grad = sqrt((xd.^2)+(yd.^2));
    wdg=depth_grad.*double(hed);
    point_map=wdg;
    point_map(point_map < 30*mean(point_map(:))) = 0;
    wdg(wdg<20*mean(wdg(:)))=0;wdg(wdg>0)=1;
    gausFilter = fspecial('gaussian',80,25);
    blur=imfilter(wdg,gausFilter,'replicate'); 
    gausFilter = fspecial('gaussian',20,20);
    blur=imfilter(blur*10,gausFilter,'replicate');
    result = cal_smooth_return_adaptive(rgb,blur,point_map);
    imwrite(result,[dir_RQ list(i).name]);

    %% SM   
    [rx,ry]=gradient(double(rgb(:,:,1)));
    [gx,gy]=gradient(double(rgb(:,:,2)));
    [bx,by]=gradient(double(rgb(:,:,3)));
    a=sqrt((rx.^2)+(ry.^2));
    b=sqrt((gx.^2)+(gy.^2));
    c=sqrt((bx.^2)+(by.^2));
    rgb_grad = (a+b+c)/3;

    rgb_bw = rgb_grad; rgb_bw(rgb_bw<2*mean(rgb_bw(:)))=0;rgb_bw(rgb_bw>0)=1;
    depth_bw=depth_grad;depth_bw(depth_bw<5*mean(depth_bw(:)))=0;depth_bw(depth_bw>0)=1;
    %% Sturcture_similarity
    choose_1=rgb_grad.*depth_grad;
    choose_2=rgb_grad.^2+depth_grad.^2;
    choose_re1=1 - (choose_1 + 100)./(choose_2 + 100);
    %% Entropy map
    rgb1=prog_tr(rgb); 
    imwrite(rgb1/max(rgb1(:)),['./t123/rgb_' str]);
    depth1=prog_tr(depth);
    imwrite(depth1/max(depth1(:)),['./t123/depth_' str]);

    entr = imread(['./t123/rgb_' str])-imread(['./t123/depth_' str]);

    [labels, numlabels] = slicmex(rgb,400,5);labels=labels+1;
    mid_sp=[];
    for j=1:numlabels
        [tr,tc]=find(labels==j);
        mid_p=[0.5*(min(tr)+max(tr)),0.5*(min(tc)+max(tc))];
        mid_sp=[mid_sp;mid_p];
    end
    [tr,tc]=find(depth_bw>0);eff_point=[];
    row1=tr-3;row2=tr+3;
    row1(row1<1)=1;row2(row2>size(rgb,1))=size(rgb,1);
    col1=tc-3;col2=tc+3;
    col1(col1<1)=1;col2(col2>size(rgb,2))=size(rgb,2);
    for j=1:length(tr)
        box1=ones(row2(j)-row1(j)+1,col2(j)-col1(j)+1);
        box2=rgb_bw(row1(j):row2(j),col1(j):col2(j));
        box=box1-box2;t=min(box(:));
        if t == 0
            eff_point=[eff_point;j];
        end
    end
    eff_map=zeros(size(depth,1),size(depth,2));
    eff_map(sub2ind(size(depth_bw),tr(eff_point),tc(eff_point)))=1;

    %% Prior Map
    eucl_map=zeros(size(rgb,1),size(rgb,2));
    for j=1:numlabels
        % eucl_map
        t1=[tr(eff_point),tc(eff_point)];
        t2=zeros(length(eff_point),2)+mid_sp(j,:);
        t3=t1-t2;
        t3=t3.*t3;
        t3=sqrt(t3(:,1)+t3(:,2));
        eucl_map(labels==j)=sum(t3);
    end
    eucl_map=(eucl_map-min(eucl_map(:)))/(max(eucl_map(:))-min(eucl_map(:)));
    eucl_map = exp(-eucl_map*7);eucl_map=(eucl_map-min(eucl_map(:)))/(max(eucl_map(:))-min(eucl_map(:)));

    feature1 = double(entr).*eucl_map;
    feature1 = imresize(feature1,[300,300]);
    choose_re1 = imresize(choose_re1,[300,300]);
    rgb = imresize(rgb,[300,300]);

    result1 = zeros(300,300);
    result2 = zeros(300,300);
    [labels, numlabels] = slicmex(rgb,1000,5);labels=labels+1;
    for j=1:numlabels
        [tx,ty] = find(labels == j);
        result1(sub2ind(size(result1),tx,ty)) = mean(feature1(sub2ind(size(feature1),tx,ty)));
        result2(sub2ind(size(result2),tx,ty)) = mean(choose_re1(sub2ind(size(choose_re1),tx,ty)));
    end

    result1 = imresize(result1,[size(depth,1),size(depth,2)]);
    result2 = imresize(result2,[size(depth,1),size(depth,2)]);

    result2 = (result2-min(result2(:)))/(max(result2(:))-min(result2(:)));
    re = result1.*(result2*0.3 + 0.7);
    rgb = imread([dir_rgb list(i).name]);
    imwrite(cal_smooth_return( rgb,re,600),[dir_SM list(i).name]);

%     %% SMM feature
%     rgbd=im2double(imread([dir_rgbd str(1:end-3) 'png']));
%     rgbdrand=im2double(imread([dir_rgbd_rand str(1:end-3) 'png']));
% 
%     tmp=rgbd-rgbdrand;
%     tmp=abs(tmp);
%     % smooth
%     imwrite(cal_smooth_return(rgb,tmp,600),[dir_SMM str(1:end-3) 'png']);
end
rmdir('./t123/','s');