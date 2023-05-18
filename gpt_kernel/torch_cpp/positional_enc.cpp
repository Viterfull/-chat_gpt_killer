#include "positional_enc.hpp"

torch::Tensor PositionalEncodingImpl::forward() {

    torch::Tensor even_i = torch::arange(0, d_model, 2, torch::kFloat);
    torch::Tensor denominator = torch::pow(10000, even_i / d_model);
    torch::Tensor position = torch::arange(max_sequence_length).reshape({max_sequence_length, 1});

    torch::Tensor even_PE = torch::sin(position / denominator);
    torch::Tensor odd_PE = torch::cos(position / denominator);
    torch::Tensor stacked = torch::stack({even_PE, odd_PE}, /*dim=*/2);

    torch::Tensor PE = torch::flatten(stacked, /*start_dim=*/1, /*end_dim=*/2);

    return PE;
    
}