#pragma once

#include <torch/torch.h>
#include <utils.hpp>

class MultiHeadCrossAttention : public torch::nn::Module {
public:
    MultiHeadCrossAttention(int d_model, int num_heads);

    torch::Tensor forward(torch::Tensor x, torch::Tensor y, torch::Tensor mask);

private:
    int d_model_;
    int num_heads_;
    int head_dim_;
    torch::nn::Linear kv_layer_;
    torch::nn::Linear q_layer_;
    torch::nn::Linear linear_layer_;
};

TORCH_MODULE(MultiHeadCrossAttention)