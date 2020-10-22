#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "dataset.h"
#include "alexnet.h"

using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using torch::Tensor;
using torch::indexing::Slice;
using torch::data::make_data_loader;

int main() {
    cout << "AlexNet training on Cifar..." << endl;

    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Using GPU." : "CUDA not available. Using CPU.") << '\n';

    // Hyper parameters
    int64_t num_classes = 10;
    int64_t batch_size = 32;
    size_t num_epochs = 40;
    double learning_rate = 1e-3;

    string path = "cifar-10-batches-bin/";


    // CIFAR10 dataset
    auto tStart = std::chrono::high_resolution_clock::now();
    auto train_dataset = Cifar(path, true)
                            .map(torch::data::transforms::Stack<>());    

    // Test set
    auto test_dataset = Cifar(path, false)
                            .map(torch::data::transforms::Stack<>());    

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( tEnd - tStart ).count();
    cout << "Loaded datasets. It took " << duration*1e-6 << " seconds" <<endl;

    // Number of samples in the training set
    size_t num_train_samples = train_dataset.size().value();
    cout << "num_train_samples: " << num_train_samples << endl;

    size_t num_test_samples = test_dataset.size().value();
    cout << "num_test_samples: " << num_test_samples << endl;

    // Data loader
    auto train_loader = make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset), batch_size);

    AlexNet model(num_classes);
    model.to(device);

    // Optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));

    cout << fixed << std::setprecision(4); 

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto batch : *train_loader) {
            optimizer.zero_grad();

            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            auto output = model.forward(data);

            auto loss = torch::nn::functional::cross_entropy(output, target);

            running_loss += loss.item<double>() * data.size(0);

            auto prediction = output.argmax(1);

            num_correct += prediction.eq(target).sum().item<int64_t>();

            loss.backward();
            optimizer.step();

        }
        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        cout << "Epoch [" << (epoch + 1)
             << "/" << num_epochs 
             << "], Trainset - Loss: "
             << sample_mean_loss << ", Accuracy: " 
             << accuracy << '\n';
    }

    cout << "Testing...\n";

    // Test the model
    model.eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        auto output = model.forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }
    
    cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    cout << "Testset - Loss: " 
         << test_sample_mean_loss 
         << ", Accuracy: " 
         << test_accuracy << '\n';
    
    return 0;
}