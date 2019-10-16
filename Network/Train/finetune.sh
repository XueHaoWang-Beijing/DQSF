sudo /home/wang/Documents/caffe/build/tools/caffe train -solver solver.prototxt -weights ./modify/net_after.caffemodel -gpu 0 2>&1 | tee ./snapshot/log/log.log
