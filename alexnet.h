#include <torch/torch.h>
#include <torch/nn/modules/dropout.h>
#include <vector>

using torch::Tensor;

class AlexNet : public torch::nn::Module {
    // Modified AlexNet for Cifar dataset
    public:
        explicit AlexNet(int64_t num_classes=100);
        Tensor forward(Tensor x);

    private:
        torch::nn::Sequential features{
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(2).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),

            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}))
        };

        torch::nn::Sequential classifier{
            torch::nn::Dropout(torch::nn::Dropout2dOptions().p(0.5)),
            torch::nn::Linear(torch::nn::LinearOptions(256 * 2 * 2, 4096)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Dropout(torch::nn::Dropout2dOptions().p(0.5)),
            torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))
        };

        torch::nn::Linear fc;
};



class AlexNet_ILSVRC2012 : public torch::nn::Module {
    public:
        explicit AlexNet_ILSVRC2012(int64_t num_classes=1000);
        Tensor forward(Tensor x);

    private:
        torch::nn::Sequential features{
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3, 2})),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3, 2})),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),

            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),

            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3, 2}))
        };

        torch::nn::Sequential classifier{
            torch::nn::Dropout(torch::nn::Dropout2dOptions().p(0.5)),
            torch::nn::Linear(torch::nn::LinearOptions(256 * 6 * 6, 4096)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Dropout(torch::nn::Dropout2dOptions().p(0.5)),
            torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))
        };

        torch::nn::Linear fc;
};