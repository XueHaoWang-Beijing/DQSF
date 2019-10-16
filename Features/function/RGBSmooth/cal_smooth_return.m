function temp = cal_smooth_return( rgb,sal,num)


% addpath('/media/wang/1T/Fdisk/function/SLIC/CCC-cuda/');

    im=double(sal);
    img=rgb;
    dd=300;
    w=size(img,1); h=size(img,2);
    img = imresize(img,[dd,dd]);
    uu = im;
    uu = imresize(uu,[dd,dd]);
    uu = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));

    temp0 = RGBSmooth(img,uu,50,0.5,num);
    temp0 = (temp0-min(min(temp0)))/(max(max(temp0))-min(min(temp0)));
    temp1 = RGBSmooth(img,uu,50,0.5,num-300);
    temp1 = (temp1-min(min(temp1)))/(max(max(temp1))-min(min(temp1)));
    temp2 = RGBSmooth(img,uu,50,0.5,num-200);
    temp2 = (temp2-min(min(temp2)))/(max(max(temp2))-min(min(temp2)));
    temp3 = RGBSmooth(img,uu,50,0.5,num-100);
    temp3 = (temp3-min(min(temp3)))/(max(max(temp3))-min(min(temp3)));

    temp = temp0+temp1+temp2+temp3;
    if(max(max(uu))==0)
        temp = zeros(dd,dd);
    end
    temp=temp/4;
    temp=imresize(temp,[w,h]);
end
