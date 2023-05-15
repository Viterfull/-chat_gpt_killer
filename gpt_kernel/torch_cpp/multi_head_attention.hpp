#pragma once

#include <torch/torch.h>
#include <utils.hpp>

class MultiHeadAttention : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int64_t d_model, int64_t num_heads);

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask);

private:
    int64_t d_model;
    int64_t num_heads;
    int64_t head_dim;
    torch::nn::Linear qkv_layer;
    torch::nn::LayerNorm ln;
};

TORCH_MODULE(MultiHeadAttention);