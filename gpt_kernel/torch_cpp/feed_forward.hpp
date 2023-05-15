#pragma once

#include <torch/torch.h>
#include <utils.hpp>

class FeedFoward : public torch::nn::Module {
public:
    FeedFowardImpl(int64_t d_model, int64_t hidden, double drop_prob = 0.2);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential net;
};

TORCH_MODULE(FeedFoward);