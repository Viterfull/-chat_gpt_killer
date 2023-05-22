#include "decoder_layer.hpp"

DecoderLayerImpl::DecoderLayerImpl(int64_t d_model, int ffn_hidden, int num_heads, float drop_prob)
    : self_attention_(d_model, num_heads),
      layer_norm1_(torch::nn::LayerNormOptions({d_model})),
      dropout1_(drop_prob),
      encoder_decoder_attention_(d_model, num_heads),
      layer_norm2_(torch::nn::LayerNormOptions({d_model})),
      dropout2_(drop_prob),
      ffn_(d_model, ffn_hidden, drop_prob),
      layer_norm3_(torch::nn::LayerNormOptions({d_model})),
      dropout3_(drop_prob) {}

torch::Tensor DecoderLayerImpl::forward(torch::Tensor x, torch::Tensor y, torch::Tensor self_attention_mask, torch::Tensor cross_attention_mask) {
    
    torch::Tensor _y = y.clone();
    y = self_attention_.forward(y, self_attention_mask);
    y = dropout1_->forward(y);
    y = layer_norm1_->forward(y + _y);

    _y = y.clone();
    y = encoder_decoder_attention_.forward(x, y, cross_attention_mask);
    y = dropout2_->forward(y);
    y = layer_norm2_->forward(y + _y);

    _y = y.clone();
    y = ffn_.forward(y);
    y = dropout3_->forward(y);
    y = layer_norm3_->forward(y + _y);

    return y;
}
