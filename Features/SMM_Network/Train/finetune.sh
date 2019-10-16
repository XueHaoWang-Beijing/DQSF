sudo /home/wang/Documents/caffe/build/tools/caffe train -solver solver_rgbd.prototxt -weights ./modify/net_after.caffemodel -gpu 0 2>&1 | tee ./snapshot/log/logRGBD.log
sudo /home/wang/Documents/caffe/build/tools/caffe train -solver solver_rgbdr.prototxt -weights ./modify/net_after.caffemodel -gpu 0 2>&1 | tee ./snapshot/log/logRGBDr.log
