#pragma once

#include <torch/torch.h>
#include "sentence_emb.hpp"
#include "sequential_encoder.hpp"
#include "encoder_layer.hpp"

class EncoderImpl : public torch::nn::Module {

public:
    EncoderImpl(int d_model, int ffn_hidden, int num_heads, float drop_prob,
            int num_layers, int max_sequence_length, std::unordered_map<std::string, int> language_to_index,
            std::string START_TOKEN, std::string END_TOKEN, std::string PADDING_TOKEN);

    torch::Tensor forward(torch::Tensor x, torch::Tensor self_attention_mask,
                          std::string start_token, std::string end_token);

private:
    SentenceEmbedding sentence_embedding_{};
    SequentialEncoder layers_{};
    
};

TORCH_MODULE(Encoder);