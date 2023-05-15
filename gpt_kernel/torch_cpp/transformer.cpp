#include "transformer.hpp"

TransformerImpl::Transformer(int d_model, 
                                 int ffn_hidden, 
                                 int num_heads, 
                                 float drop_prob, 
                                 int num_layers,
                                 int max_sequence_length, 
                                 int trg_vocab_size,
                                 std::unordered_map<std::string, int> src_to_index,
                                 std::unordered_map<std::string, int> trg_to_index,
                                 std::unordered_map<int, std::string> index_to_trg,
                                 std::string START_TOKEN, 
                                 std::string END_TOKEN, 
                                 std::string PADDING_TOKEN
                                 )
    : max_sequence_length_(max_sequence_length),
      trg_vocab_size_(trg_vocab_size),
      trg_to_index_(trg_to_index),
      index_to_trg_(index_to_trg),
      PADDING_TOKEN_(PADDING_TOKEN),
      START_TOKEN_(START_TOKEN),
      END_TOKEN_(END_TOKEN),
      encoder_(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, src_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN),
      decoder_(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, trg_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN),
      linear_(torch::nn::LinearOptions(d_model, trg_vocab_size)),
      device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {

    register_module("encoder", encoder_);
    register_module("decoder", decoder_);
    register_module("linear", linear_);
}

torch::Tensor Transformer::forward(torch::Tensor x, 
                                    torch::Tensor y, 
                                    torch::Tensor encoder_self_attention_mask,
                                    torch::Tensor decoder_self_attention_mask,
                                    torch::Tensor decoder_cross_attention_mask,
                                    bool enc_start_token,
                                    bool enc_end_token,
                                    bool dec_start_token, 
                                    bool dec_end_token) {
    x = encoder_->forward(x, encoder_self_attention_mask, enc_start_token, enc_end_token);
    torch::Tensor out = decoder_->forward(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, dec_start_token, dec_end_token);
    out = linear_->forward(out);

    return out;
}

std::string TransformerImpl::translate(std::string src_sentence) {

    eval();
    int max_sequence_length = max_sequence_length_;
    torch::Device device = device_;
    std::vector<std::string> src_sentence_vec = {src_sentence};
    std::vector<std::string> trg_sentence_vec = {""};
    
    for (int word_counter = 0; word_counter < max_sequence_length; word_counter++) {
        
        torch::Tensor encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask;
        std::tie(encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask) = create_masks(src_sentence_vec, trg_sentence_vec);
        torch::Tensor predictions = forward(torch::tensor(src_sentence_vec), 
                                             torch::tensor(trg_sentence_vec), 
                                             encoder_self_attention_mask.to(device), 
                                             decoder_self_attention_mask.to(device), 
                                             decoder_cross_attention_mask.to(device),
                                             false,
                                             false,
                                             true,
                                             false);
        torch::Tensor next_token_prob_distribution = predictions[0][word_counter];
        int next_token_index = at::argmax(next_token_prob_distribution).item<int>();
        std::string next_token = index_to_trg_[next_token_index];

        if (next_token == END_TOKEN_) {
            break;
        }

        trg_sentence_vec[0] += next_token;
    }
    return trg_sentence_vec[0];
}