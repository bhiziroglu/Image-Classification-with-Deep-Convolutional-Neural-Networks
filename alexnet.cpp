#include <torch/torch.h>
#include <vector>
#include "alexnet.h"

using torch::Tensor;
using std::cout;
using std::endl;

AlexNet::AlexNet(int64_t num_classes) : fc(4096, num_classes){
    register_module("features", features);
    register_module("classifier", classifier);
    register_module("fc", fc);
}

Tensor AlexNet::forward(torch::Tensor x){
    x = features->forward(x);
    x = x.view({-1, 1024});
    x = classifier->forward(x);
    x = fc->forward(x);
    return x;
}