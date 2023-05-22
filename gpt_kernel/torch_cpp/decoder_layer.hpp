#pragma once

#include <torch/torch.h>
#include "multi_head_attention.hpp"
#include "multi_head_cross_attention.hpp"
#include "feed_forward.hpp"

class DecoderLayerImpl : public torch::nn::Module {
public:
    DecoderLayerImpl(int64_t d_model, int ffn_hidden, int num_heads, float drop_prob);

    torch::Tensor forward(torch::Tensor x, torch::Tensor y, torch::Tensor self_attention_mask, torch::Tensor cross_attention_mask);

private:
    MultiHeadAttentionImpl self_attention_;
    torch::nn::LayerNorm layer_norm1_;
    torch::nn::Dropout dropout1_;
    MultiHeadCrossAttentionImpl encoder_decoder_attention_;
    torch::nn::LayerNorm layer_norm2_;
    torch::nn::Dropout dropout2_;
    FeedForwardImpl ffn_;
    torch::nn::LayerNorm layer_norm3_;
    torch::nn::Dropout dropout3_;
};

TORCH_MODULE(DecoderLayer);