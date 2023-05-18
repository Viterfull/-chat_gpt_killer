#pragma once

#include <torch/torch.h>

class PositionalEncodingImpl : public torch::nn::Module {

    public:
        PositionalEncodingImpl(int64_t d_model, int64_t max_sequence_length)
            : max_sequence_length(max_sequence_length), d_model(d_model){}

        torch::Tensor forward();

    private:
        int64_t max_sequence_length;
        int64_t d_model;
};

TORCH_MODULE(PositionalEncoding);