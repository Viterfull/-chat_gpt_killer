#include "multi_head_cross_attention.hpp"

MultiHeadCrossAttentionImpl::MultiHeadCrossAttentionImpl(int d_model, int num_heads)
    : d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(d_model / num_heads),
      kv_layer_(d_model, 2 * d_model),
      q_layer_(d_model, d_model),
      linear_layer_(d_model, d_model) {}

torch::Tensor MultiHeadCrossAttentionImpl::forward(torch::Tensor x, torch::Tensor y, torch::Tensor mask) {
    int batch_size = x.size(0);
    int sequence_length = x.size(1);
    torch::Tensor kv = kv_layer_->forward(x);
    torch::Tensor q = q_layer_->forward(y);

    kv = kv.view({batch_size, sequence_length, num_heads_, 2 * head_dim_});
    q = q.view({batch_size, sequence_length, num_heads_, head_dim_});
    kv = kv.permute({0, 2, 1, 3});
    q = q.permute({0, 2, 1, 3});
    torch::Tensor k, v;
    std::tie(k, v) = kv.chunk(2, -1);

    torch::Tensor values;
    values = scaled_dot_product(q, k, v, mask);
    values = values.permute({0, 2, 1, 3}).reshape({batch_size, sequence_length, d_model_});

    torch::Tensor out = linear_layer_->forward(values);

    return out;
}
