#include "dataset.h"

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>

using std::pair;

string Cifar::join_paths(string head, const string &tail)
    {
        if (head.back() != '/')
        {
            head.push_back('/');
        }
        head += tail;
        return head;
    }

pair<Tensor, Tensor> Cifar::read_data(const string &root, bool train){
    vector<string> files;
    int num_samples;

    if(train==true){
        files = trainBatchFiles;
        num_samples = 50000;
    }else{
        files = testBatchFiles;
        num_samples = 10000;
    }

    vector<char> data_buffer;
    data_buffer.reserve(files.size() * 30730000);

    for (const auto &file : files)
    {
        const auto path = join_paths(root, file);
        ifstream data(path, std::ios::binary);
        TORCH_CHECK(data, "Error opening data file at", path);

        data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});
    }

    TORCH_CHECK(data_buffer.size() == files.size() * 30730000, "Unexpected file sizes");

    auto images = torch::empty({num_samples, 3, 32, 32}, torch::kByte);
    auto targets = torch::empty(num_samples, torch::kByte);
    
    for (uint32_t i = 0; i != num_samples; ++i)
    {
        // The first byte of each row is the target class index.
        uint32_t start_index = i * 3073;
        targets[i] = data_buffer[start_index];

        // The next bytes correspond to the rgb channel values in the following order:
        // red (32 *32 = 1024 bytes) | green (1024 bytes) | blue (1024 bytes)
        uint32_t image_start = start_index + 1;
        uint32_t image_end = image_start + 3 * 1024;
        std::copy(data_buffer.begin() + image_start, data_buffer.begin() + image_end,
                    reinterpret_cast<char *>(images[i].data_ptr()));
    }

    return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
}