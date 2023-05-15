#pragma once

#include <torch/torch.h>
#include "sentence_emb.hpp"
#include "sequential_decoder.hpp"
#include "decode_layer.hpp"

class Decoder : public torch::nn::Module {
public:
    DecoderImpl(int64_t d_model, int64_t ffn_hidden, int64_t num_heads, double drop_prob, int64_t num_layers,
        int64_t max_sequence_length, const std::unordered_map<std::string, int64_t>& language_to_index, int64_t START_TOKEN,
        int64_t END_TOKEN, int64_t PADDING_TOKEN);

    torch::Tensor forward(torch::Tensor x, torch::Tensor y, torch::Tensor self_attention_mask, torch::Tensor cross_attention_mask,
        int64_t start_token, int64_t end_token);

private:
    std::vector<DecoderLayer> create_decoder_layers(int64_t d_model, int64_t ffn_hidden, int64_t num_heads, double drop_prob, int64_t num_layers);

    SentenceEmbedding sentence_embedding{ nullptr };
    SequentialDecoder layers{ nullptr };
};

TORCH_MODULE(Decoder);
