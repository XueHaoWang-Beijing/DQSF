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
for iter = 15000:-5000:15000
    
    net = caffe.Net('./deploy.prototxt', ['./RGBD_iter_' num2str(iter) '.caffemodel'], 'test');
    netr = caffe.Net('./deploy.prototxt', ['./RGBD_iter_' num2str(iter) '.caffemodel'], 'test');

    dir_Dataset = '/home/wang/Documents/RGBDset/3Test/';
    dir_Result = ['./prediction' num2str(iter) '/'];
    Dataset = {'NJUDS','NLPR','STERE','SSD','LFSD','DES'};
    randD = imread('/home/wang/Documents/RGBDset/random_depth.jpg');

    for d = 1:length(Dataset)

        ds = Dataset{d};
        dir_rgb = [dir_Dataset Dataset{d} '/RGB/'];
        dir_depth = [dir_Dataset Dataset{d} '/depth/'];
        
        mkdir(['../' Dataset{d} '/SSM/']);
        mkdir(['../' Dataset{d} '/SSMs/']);
        mkdir(['../' Dataset{d} '/SSMsr/']);

        list = dir([dir_rgb '*.jpg']);
        for i=1:length(list)
            disp([ds ': ' num2str(i)]);
            str = list(i).name;
            rgb = imread([dir_rgb str]);
            depth = imread([dir_depth str(1:end-3) 'png']);
            depthr = imresize(randD,[224,224]);
            
            [m,n,~] = size(rgb);

            rgb = single(imresize(rgb,[224,224]));
            depth = single(imresize(depth,[224,224]));
            % RGB
            data = zeros(224,224,3,1,'single');
            data(:,:,1,1) = (rgb(:,:,3)-102.111398)*0.00390625;
            data(:,:,2,1) = (rgb(:,:,2)-109.662439)*0.00390625;
            data(:,:,3,1) = (rgb(:,:,1)-112.768360)*0.00390625;
            data=permute(data,[2 1 3 4]);
            net.blobs('rgb').set_data(data);
            netr.blobs('rgb').set_data(data);

            % Depth
            data = zeros(224,224,1,1,'single');
            data(:,:,1,1) = (depth-127.840869)*0.00390625;
            data=permute(data,[2 1 3 4]);
            net.blobs('depth').set_data(data);
            
            %Depthr
            data = zeros(224,224,1,1,'single');
            data(:,:,1,1) = (depthr-mean2(depthr))*0.00390625;
            data=permute(data,[2 1 3 4]);
            netr.blobs('depth').set_data(data);
            
            net.forward_prefilled();
            netr.forward_prefilled();
            
            csal = net.blobs('rgb_sal8').get_data();
            uu = csal';
            uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
            imwrite(imresize(uu1,[m,n]),['../' Dataset{d} '/SSMs/' list(i).name(1:end-3) 'png']);
            
            csal = netr.blobs('rgb_sal8').get_data();
            uu = csal';
            uu2 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
            imwrite(imresize(uu2,[m,n]),['../' Dataset{d} '/SSMsr/' list(i).name(1:end-3) 'png']);
            
            sal = abs(uu2-uu1);
            sal = (sal-min(min(sal)))/(max(max(sal))-min(min(sal)));
            imwrite(imresize(sal,[m,n]),['../' Dataset{d} '/SSM/' list(i).name(1:end-3) 'png']);
        end
    end
end
caffe.reset_all();

