#pragma once

#include <torch/torch.h>
#include "decoder_layer.hpp"

class SequentialDecoder : public torch::nn::Sequential {

public:
    torch::Tensor forward(torch::Tensor& x, torch::Tensor& y, torch::Tensor& self_attention_mask, torch::Tensor& cross_attention_mask);
    
};
