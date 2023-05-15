#include "multi_head_attention.hpp"

MultiHeadAttentionImpl::MultiHeadAttention(int64_t d_model, int64_t num_heads)
    : d_model(d_model), num_heads(num_heads) {

    head_dim = d_model / num_heads;
    qkv_layer = register_module("qkv_layer", torch::nn::Linear(d_model, 3 * d_model));
    ln = register_module("ln", torch::nn::LayerNorm(d_model));

}

torch::Tensor MultiHeadAttention::forward(torch::Tensor x, torch::Tensor mask) {

    auto sizes = x.sizes();
    int64_t B = sizes[0];
    int64_t T = sizes[1];
    int64_t C = sizes[2];

    auto qkv = qkv_layer->forward(x);
    qkv = qkv.reshape({B, T, num_heads, 3 * head_dim});
    qkv = qkv.permute({0, 2, 1, 3});

    auto q = qkv.select(3, 0);
    auto k = qkv.select(3, head_dim);
    auto v = qkv.select(3, 2 * head_dim);
    auto values = scaled_dot_product(q, k, v, mask);

    values = values.permute({0, 2, 1, 3}).reshape({B, T, num_heads * head_dim});
    auto out = ln->forward(values);

    return out;

}
