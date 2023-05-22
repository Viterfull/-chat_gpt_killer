#include "sentence_emb.hpp"

torch::Tensor SentenceEmbeddingImpl::batch_tokenize(const std::vector<std::string>& batch, bool start_token, bool end_token) {
    std::vector<torch::Tensor> tokenized;

    for (const auto& sentence : batch) {
        
        std::vector<int64_t> sentence_word_indicies;
        
        for (const auto& token : sentence) {
            sentence_word_indicies.push_back(language_to_index_.at(std::string(1, token)));
        }

        if (start_token) {
            sentence_word_indicies.insert(sentence_word_indicies.begin(), language_to_index_.at(START_TOKEN_));
        }
        if (end_token) {
            sentence_word_indicies.push_back(language_to_index_.at(END_TOKEN_));
        }
        for (size_t i = sentence_word_indicies.size(); i < static_cast<size_t>(max_sequence_length_); ++i) {
            sentence_word_indicies.push_back(language_to_index_.at(PADDING_TOKEN_));
        }
        auto tensor = torch::tensor(sentence_word_indicies, torch::kLong);
        tokenized.push_back(tensor);
    }
        
    auto tokenized_tensor = torch::stack(tokenized);
    return tokenized_tensor.to(at::kCUDA);

}

torch::Tensor SentenceEmbeddingImpl::forward(const std::vector<std::string>& x, bool start_token, bool end_token) {

    auto tokenized = batch_tokenize(x, start_token, end_token);
    auto embedded = embedding_.forward(tokenized);
    auto pos = position_encoder_.forward().to(at::kCUDA)
    auto dropout_output = dropout_.forward(embedded + pos);

    return dropout_output;
}