#pragma once

#include <torch/torch.h>
#include "encoder_layer.hpp"

class SequentialEncoder : public torch::nn::Sequential {
public:
    torch::Tensor forward(torch::Tensor x, torch::Tensor self_attention_mask);
};