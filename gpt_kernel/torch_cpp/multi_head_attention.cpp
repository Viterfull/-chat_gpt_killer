#include "multi_head_attention.hpp"

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t d_model, int64_t num_heads)
    : d_model(d_model), num_heads(num_heads) {

    head_dim = d_model / num_heads;
    qkv_layer = register_module("qkv_layer", torch::nn::Linear(d_model, 3 * d_model));
    ln = register_module("ln", torch::nn::LayerNorm(d_model));

}
