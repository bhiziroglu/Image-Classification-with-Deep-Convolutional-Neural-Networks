#include <torch/torch.h>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::move;
using std::ifstream;
using std::string;
using std::vector;
using std::pair;
using torch::Tensor;
using torch::tensor;
using torch::data::datasets::Dataset;

using dtype = torch::data::Example<>;
class Cifar : public Dataset<Cifar, dtype> //Cifar10 only

{
private:
    torch::Tensor images_;
    torch::Tensor targets_;

    vector<string> trainBatchFiles = {"data_batch_1.bin",
                                                 "data_batch_2.bin",
                                                 "data_batch_3.bin",
                                                 "data_batch_4.bin",
                                                 "data_batch_5.bin"};

    vector<string> testBatchFiles = {"test_batch.bin"};

    string join_paths(string head, const string &tail);

    pair<Tensor, Tensor> read_data(const string &root, bool train);

public:
    // Constructor
    Cifar(const string& root, bool train){
        auto data = read_data(root, train);
        images_ = move(data.first);
        targets_ = move(data.second);
    };

    // Override get() function to return tensor at location index
    virtual dtype get(size_t index) override
    {
        return {images_[index], targets_[index]};
    };

    // Return the length of data
    virtual torch::optional<size_t> size() const override
    {
        return images_.size(0);
    };
};