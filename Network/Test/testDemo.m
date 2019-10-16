clc; clear all; close all;

addpath(genpath('/home/wang/Documents/caffe/matlab/'));

caffe.reset_all();
use_gpu=1;
gpu_id=0;
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
for iter = 6000:-1000:5500
    net = caffe.Net('../deploy.prototxt', ['../snapshot/ALL_iter_' num2str(iter) '.caffemodel'], 'test');

    dir_Dataset = '/home/wang/Documents/RGBDset/3Test/';
    dir_Result = ['./prediction' num2str(iter) '/'];
    % Dataset = {'NJUDS','NLPR','STERE','SSD','LFSD','DES'};
    Dataset = {'NLPR','STERE','SSD','LFSD','DES'};

    for d = 4:length(Dataset)

        ds = Dataset{d};
        dir_rgb = [dir_Dataset ds '/RGB/'];
        dir_cpfp = ['/home/wang/Documents/RGBDset/Dataset/Additional/CPFP/' Dataset{d} '/'];
        dir_poolnet = ['/home/wang/Documents/RGBDset/Dataset/Additional/PoolNet/' Dataset{d} '/'];
        dir_RQ = ['/home/wang/Documents/RGBD328/20190908/FeaturesAll/' Dataset{d} '/RQ/'];
        dir_SM = ['/home/wang/Documents/RGBD328/20190908/FeaturesAll/' Dataset{d} '/SM/'];
        dir_SMM = ['/home/wang/Documents/RGBD328/20190908/FeaturesAll/' Dataset{d} '/SSM/'];

%         mkdir([dir_Result ds '/rgb/']);
        mkdir([dir_Result ds '/f1/']);
        mkdir([dir_Result ds '/f2/']);
        mkdir([dir_Result ds '/f3/']);
        mkdir([dir_Result ds '/fuse/']);

        list = dir([dir_rgb '*.jpg']);
        for i=1:length(list)
            disp([ds ': ' num2str(i)]);
            str = list(i).name;
            rgb = imread([dir_rgb str]);
            cpfp = imread([dir_cpfp str(1:end-3) 'png']);
            poolnet = imread([dir_poolnet str(1:end-3) 'png']);
            rq = imread([dir_RQ str(1:end-3) 'png']);
            sm = imread([dir_SM str(1:end-3) 'png']);
            ssm = imread([dir_SMM str(1:end-3) 'png']);
            
            [m,n,~] = size(rgb);

            rgb = single(imresize(rgb,[224,224]));
            cpfp = single(imresize(cpfp,[224,224]));
            poolnet = single(imresize(poolnet,[224,224]));
            rq = single(imresize(rq,[224,224]));
            sm = single(imresize(sm,[224,224]));
            ssm = single(imresize(ssm,[224,224]));
%             % RGB
%             data = zeros(224,224,3,1,'single');
%             data(:,:,1,1) = (rgb(:,:,3)-102.111398)*0.00390625;
%             data(:,:,2,1) = (rgb(:,:,2)-109.662439)*0.00390625;
%             data(:,:,3,1) = (rgb(:,:,1)-112.768360)*0.00390625;
%             data=permute(data,[2 1 3 4]);
%             net.blobs('rgb').set_data(data);

            % CPFP
            data = zeros(224,224,1,1,'single');
            data(:,:,1,1) = cpfp*0.00390625;
            data=permute(data,[2 1 3 4]);
            net.blobs('cpfp').set_data(data);
            
            % PoolNet
            data = zeros(224,224,1,1,'single');
            data(:,:,1,1) = poolnet*0.00390625;
            data=permute(data,[2 1 3 4]);
            net.blobs('poolnet').set_data(data);
            
            % RQ
            data = zeros(224,224,1,1,'single');
            data(:,:,1,1) = rq*0.00390625;
            data=permute(data,[2 1 3 4]);
            net.blobs('feature1').set_data(data);
            
            % SM
            data = zeros(224,224,1,1,'single');
            data(:,:,1,1) = sm*0.00390625;
            data=permute(data,[2 1 3 4]);
            net.blobs('feature2').set_data(data);
            
            % SSM
            data = zeros(224,224,1,1,'single');
            data(:,:,1,1) = ssm*0.00390625;
            data=permute(data,[2 1 3 4]);
            net.blobs('feature3').set_data(data);
            
            net.forward_prefilled();
            
%             csal = net.blobs('rgb_sal8').get_data();
%             uu = csal';
%             uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
%             imwrite(imresize(uu1,[m,n]),[dir_Result ds '/rgb/' list(i).name]);
            
            csal = net.blobs('f1_sal8').get_data();
            uu = csal';
            uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
            imwrite(imresize(uu1,[m,n]),[dir_Result ds '/f1/' list(i).name]);
            
            csal = net.blobs('f2_sal8').get_data();
            uu = csal';
            uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
            imwrite(imresize(uu1,[m,n]),[dir_Result ds '/f2/' list(i).name]);
            
            csal = net.blobs('f3_sal8').get_data();
            uu = csal';
            uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
            imwrite(imresize(uu1,[m,n]),[dir_Result ds '/f3/' list(i).name]);
            
            csal = net.blobs('sal').get_data();
            uu = csal';
            uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
            imwrite(imresize(uu1,[m,n]),[dir_Result ds '/fuse/' list(i).name]);


        end
    end
    caffe.reset_all();
end
caffe.reset_all();

