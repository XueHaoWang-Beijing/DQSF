function temp = cal_smooth_return_adaptive( rgb,sal,map)
%     addpath('/media/wang/30F89211F891D606/RGBD323/fast-rcnn/matlab/train/function/CCC-cuda/');

    im=double(sal);
    img=rgb;
    dd=300;
    w=size(img,1); h=size(img,2);
    img = imresize(img,[dd,dd]);
    map = imresize(map,[dd,dd]);
    uu = im;
    uu = imresize(uu,[dd,dd]);
    uu = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));

    para = 0.01;
    temp0 = RGBSmooth_adaptiv(img,uu,map,50,para,600);
    temp0 = (temp0-min(min(temp0)))/(max(max(temp0))-min(min(temp0)));
    temp1 = RGBSmooth_adaptiv(img,uu,map,50,para,300);
    temp1 = (temp1-min(min(temp1)))/(max(max(temp1))-min(min(temp1)));
    temp2 = RGBSmooth_adaptiv(img,uu,map,50,para,400);
    temp2 = (temp2-min(min(temp2)))/(max(max(temp2))-min(min(temp2)));
    temp3 = RGBSmooth_adaptiv(img,uu,map,50,para,500);
    temp3 = (temp3-min(min(temp3)))/(max(max(temp3))-min(min(temp3)));

    temp = temp0+temp1+temp2+temp3;
    if(max(max(uu))==0)
        temp = zeros(dd,dd);
    end
    temp=temp/4;
    temp=imresize(temp,[w,h]);
end
