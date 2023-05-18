#pragma once

#include <torch/torch.h>
#include <iostream>
#include <unordered_map>
#include <string>
#include "positional_enc.hpp"

class SentenceEmbeddingImpl : torch::nn::Module {

    public:

        SentenceEmbeddingImpl(int max_sequence_length, int d_model,
                      const std::unordered_map<std::string, int>& language_to_index,
                      const std::string& START_TOKEN,
                      const std::string& END_TOKEN,
                      const std::string& PADDING_TOKEN) :
        vocab_size_(language_to_index.size()),
        max_sequence_length_(max_sequence_length),
        embedding_(torch::nn::EmbeddingOptions(vocab_size_, d_model)),
        language_to_index_(language_to_index),
        position_encoder_(d_model, max_sequence_length),
        dropout_(torch::nn::DropoutOptions(0.1)),
        START_TOKEN_(START_TOKEN),
        END_TOKEN_(END_TOKEN),
        PADDING_TOKEN_(PADDING_TOKEN) {}

        torch::Tensor batch_tokenize(const std::vector<std::string>& batch, bool start_token, bool end_token);

        torch::Tensor forward(const std::vector<std::string>& x, bool start_token, bool end_token);

    private:

        int vocab_size_;
        int max_sequence_length_;
        torch::nn::Embedding embedding_;
        std::unordered_map<std::string, int> language_to_index_;
        PositionalEncoding position_encoder_;
        torch::nn::Dropout dropout_;
        std::string START_TOKEN_;
        std::string END_TOKEN_;
        std::string PADDING_TOKEN_;

};

TORCH_MODULE(SentenceEmbedding);