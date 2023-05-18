#include "decoder.hpp"

DecoderImpl::DecoderImpl(int64_t d_model, int64_t ffn_hidden, int64_t num_heads, double drop_prob, int64_t num_layers,
    int64_t max_sequence_length, const std::unordered_map<std::string, int64_t>& language_to_index, std::string START_TOKEN,
    std::string END_TOKEN, std::string PADDING_TOKEN)
    : sentence_embedding(register_module("sentence_embedding",
        SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)))
    , layers(register_module("layers", SequentialDecoder(create_decoder_layers(d_model, ffn_hidden, num_heads, drop_prob, num_layers))))
{}

torch::Tensor DecoderImpl::forward(torch::Tensor x, torch::Tensor y, torch::Tensor self_attention_mask, torch::Tensor cross_attention_mask,
    std::string start_token, std::string end_token) {
    y = sentence_embedding->forward(y, start_token, end_token);
    y = layers->forward(x, y, self_attention_mask, cross_attention_mask);
    return y;
}

std::vector<DecoderLayer> DecoderImpl::create_decoder_layers(int64_t d_model, int64_t ffn_hidden, int64_t num_heads, double drop_prob, int64_t num_layers) {
    
    std::vector<DecoderLayer> layers;
    layers.reserve(num_layers);

    for (int64_t i = 0; i < num_layers; ++i) {
        layers.emplace_back(DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob));
    }

    return layers;
}
