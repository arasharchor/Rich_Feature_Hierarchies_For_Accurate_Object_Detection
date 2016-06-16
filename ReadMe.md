# Object Detection using Regions with CNN features

Simple and scalable object detection algorithm by combining region proposals with CNNs to improve mean average precision on PASCAL VOC 2012 dataset using the MATLAB interface of Caffe by BVLC on Amazon’s EC2 g2.2xlarge GPU instance

##Environment:
* Caffe 0.999 (MatCaffe)
* Ubuntu 14.0.4.
* Matlab 2012b
* CUDA 7.0
* OpenBlas
* Boost 1.55
* OpenCV 2.4
* cuDNN V3

## How to run:

Run 'mycode.m' from Root-->Code-->rcnn. Edit the following parameters before running the code

* fpath --> path to store the resulting images
* rcnn_model_file --> place the R-CNN model to be used
* im on line 9 --> the path to input image