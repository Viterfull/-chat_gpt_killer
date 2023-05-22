#pragma once

#include <torch/torch.h>
#include <string>
#include "sentence_emb.hpp"
#include "sequential_decoder.hpp"
#include "decoder_layer.hpp"

class DecoderImpl : public torch::nn::Module {
public:
    DecoderImpl(int64_t d_model, int64_t ffn_hidden, int64_t num_heads, double drop_prob, int64_t num_layers,
    int64_t max_sequence_length, const std::unordered_map<std::string, int64_t>& language_to_index, std::string START_TOKEN,
    std::string END_TOKEN, std::string PADDING_TOKEN);

    torch::Tensor forward(torch::Tensor x, torch::Tensor y, torch::Tensor self_attention_mask, torch::Tensor cross_attention_mask,
        std::string start_token, std::string end_token);

private:
    std::vector<DecoderLayer> create_decoder_layers(int64_t d_model, int64_t ffn_hidden, int64_t num_heads, double drop_prob, int64_t num_layers);

    SentenceEmbedding sentence_embedding;
    SequentialDecoder layers;
};

TORCH_MODULE(Decoder);
