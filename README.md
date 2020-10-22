# Image Classification with Deep Convolutional Networks

C++ / Libtorch implementation of [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). AlexNet is the winner of 2012 ImageNet Large Scale Visual Recognition Competition.

This project implements AlexNet using C++ / Libtorch and trains it on the CIFAR dataset. 

## Requirements
- GCC / Clang
- CMake (3.10.2+)
- LibTorch (1.6.0)  
If you are going to use GPU:
- CUDA (10.0.130)
- Nvidia Driver (450.80.02)

You can install LibTorch from PyTorch's official [website](https://pytorch.org/get-started/locally/).

Download [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html).

You can use ImageNet as well. AlexNet alreadys exists [here](https://github.com/bhiziroglu/Image-Classification-with-Deep-Convolutional-Neural-Networks/blob/master/alexnet.h#L49), you would just need to write a dataloader for it.


## Clone, build and run

```
$ git clone https://github.com/bhiziroglu/Image-Classification-with-Deep-Convolutional-Neural-Networks

$ cd Image-Classification-with-Deep-Convolutional-Neural-Networks

$ mkdir build
$ cd build

$ cmake -DCMAKE_PREFIX_PATH=your_libtorch_path ..

NOTE: If you want to use GPU, you should have CUDA installed before this step. 
cmake should find your CUDA installation automatically. 
For reference, mine is installed at : /usr/local/cuda

$ cmake --build . --config Release

After you make changes to the code and want to build again:

$ make
$ ./dnn
```

Feel free to create an issue if you face build problems.

## Results

My main goal was to use C++ and Libtorch. For that reason, I didn't try to get a high test accuracy.

Test set accuracy is around 70%.  
[Current SOTA](https://benchmarks.ai/cifar-10) is 99.37%.   

You can try adding data augmentation and changing the hyperparameters to increase the test score.

### References

1) https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

2) https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp

3) https://github.com/sfd158/libtorch-dataloader-learn/blob/1ac59edf1443c447c48ce1e815236bce78d6f3d1/main.cpp

4) https://github.com/prabhuomkar/pytorch-cpp
