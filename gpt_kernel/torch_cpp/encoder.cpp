#include "encoder.hpp"

Encoder::Encoder(int d_model, int ffn_hidden, int num_heads, float drop_prob,
                 int num_layers, int max_sequence_length, std::unordered_map<std::string, int> language_to_index,
                 std::string START_TOKEN, std::string END_TOKEN, std::string PADDING_TOKEN)
    : sentence_embedding_(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN),
      layers_(num_layers) {
        
    for (int i = 0; i < num_layers; ++i) {
        layers_->push_back(std::make_shared<EncoderLayer>(d_model, ffn_hidden, num_heads, drop_prob));
    }

}

torch::Tensor Encoder::forward(torch::Tensor x, torch::Tensor self_attention_mask,
                               std::string start_token, std::string end_token) {

    x = sentence_embedding_.forward(x, start_token, end_token);
    x = layers_->forward(x, self_attention_mask);

    return x;
}
