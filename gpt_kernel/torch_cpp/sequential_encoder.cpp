#include "sequential_encoder.hpp"

torch::Tensor SequentialEncoder::forward(torch::Tensor x, torch::Tensor self_attention_mask) {

    for (const auto& module : *this) {
        
        x = module->forward(x, self_attention_mask);
    }

    return x;
}
