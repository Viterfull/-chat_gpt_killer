#include "sequential_decoder.hpp"

torch::Tensor SequentialDecoder::forward(torch::Tensor& x, torch::Tensor& y, torch::Tensor& self_attention_mask, torch::Tensor& cross_attention_mask) {

    torch::Tensor output = y;

    for (const auto& module : _modules) {
        output = module->forward(x, output, self_attention_mask, cross_attention_mask);
    }

    return output;
}
