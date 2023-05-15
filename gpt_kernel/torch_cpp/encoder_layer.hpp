#pragma once

#include <torch/torch.h>
#include "multi_head_attention.hpp"
#include "feed_forward.hpp"

class EncoderLayer : public torch::nn::Module {
public:
    EncoderLayer(int d_model, int ffn_hidden, int num_heads, double drop_prob);

    torch::Tensor forward(torch::Tensor x, torch::Tensor self_attention_mask);

private:
    MultiHeadAttention attention_;
    torch::nn::LayerNorm norm1_;
    torch::nn::Dropout dropout1_;
    FeedForward ffn_;
    torch::nn::LayerNorm norm2_;
    torch::nn::Dropout dropout2_;
};

TORCH_MODULE(EncoderLayer)