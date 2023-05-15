#include "encoder_layer.h"

EncoderLayer::EncoderLayer(int d_model, int ffn_hidden, int num_heads, double drop_prob)
    : attention_(MultiHeadAttention(d_model, num_heads)),
      norm1_(torch::nn::LayerNormOptions({d_model})),
      dropout1_(torch::nn::DropoutOptions(drop_prob)),
      ffn_(FeedForward(d_model, ffn_hidden, drop_prob)),
      norm2_(torch::nn::LayerNormOptions({d_model})),
      dropout2_(torch::nn::DropoutOptions(drop_prob)) {}

torch::Tensor EncoderLayer::forward(torch::Tensor x, torch::Tensor self_attention_mask) {
    
    torch::Tensor residual_x = x.clone();
    x = attention_.forward(x, self_attention_mask);
    x = dropout1_(x);
    x = norm1_(x + residual_x);
    residual_x = x.clone();
    x = ffn_.forward(x);
    x = dropout2_(x);
    x = norm2_(x + residual_x);
    return x;
}
