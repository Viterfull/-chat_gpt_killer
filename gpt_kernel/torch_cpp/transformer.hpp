#pragma once

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include "encoder.hpp"
#include "decoder.hpp"

class TransformerImpl : public torch::nn::Module {
public:
    TransformerImpl(int d_model, 
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
                    );
    torch::Tensor forward(torch::Tensor x, 
                          torch::Tensor y, 
                          torch::Tensor encoder_self_attention_mask = {},
                          torch::Tensor decoder_self_attention_mask = {},
                          torch::Tensor decoder_cross_attention_mask = {},
                          std::string enc_start_token = {},
                          std::string enc_end_token = {},
                          std::string dec_start_token = {}, 
                          std::string dec_end_token = {});
    std::string translate(std::string src_sentence);

private:
    int max_sequence_length_;
    int trg_vocab_size_;
    std::unordered_map<std::string, int> trg_to_index_;
    std::unordered_map<int, std::string> index_to_trg_;
    std::string PADDING_TOKEN_;
    std::string START_TOKEN_;
    std::string END_TOKEN_;
    Encoder encoder_;
    Decoder decoder_;
    torch::nn::Linear linear_;
    torch::Device device_;
};

TORCH_MODULE(Transformer);