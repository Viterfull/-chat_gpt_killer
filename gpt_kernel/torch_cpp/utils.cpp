#include "utils.hpp"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_masks(
    torch::Tensor eng_batch, torch::Tensor kn_batch) {
  
  int max_sequence_length = 200;
  float NEG_INFTY = -1e9;
  int num_sentences = eng_batch.size(0);

  torch::Tensor look_ahead_mask = torch::ones({max_sequence_length, max_sequence_length}, torch::kBool);
  look_ahead_mask = torch::triu(look_ahead_mask, 1);
  
  torch::Tensor encoder_padding_mask = torch::zeros({num_sentences, max_sequence_length, max_sequence_length}, torch::kBool);
  torch::Tensor decoder_padding_mask_self_attention = torch::zeros({num_sentences, max_sequence_length, max_sequence_length}, torch::kBool);
  torch::Tensor decoder_padding_mask_cross_attention = torch::zeros({num_sentences, max_sequence_length, max_sequence_length}, torch::kBool);

  for (int idx = 0; idx < num_sentences; idx++) {
    int eng_sentence_length = eng_batch.size(1);
    int kn_sentence_length = kn_batch.size(1);
    torch::Tensor eng_chars_to_padding_mask = torch::arange(eng_sentence_length + 1, max_sequence_length);
    torch::Tensor kn_chars_to_padding_mask = torch::arange(kn_sentence_length + 1, max_sequence_length);
    
    encoder_padding_mask[idx].index_fill_(1, eng_chars_to_padding_mask, true);
    encoder_padding_mask[idx].index_fill_(2, eng_chars_to_padding_mask, true);

    decoder_padding_mask_self_attention[idx].index_fill_(1, kn_chars_to_padding_mask, true);
    decoder_padding_mask_self_attention[idx].index_fill_(2, kn_chars_to_padding_mask, true);
    decoder_padding_mask_cross_attention[idx].index_fill_(1, eng_chars_to_padding_mask, true);
    decoder_padding_mask_cross_attention[idx].index_fill_(2, kn_chars_to_padding_mask, true);
  }

  torch::Tensor encoder_self_attention_mask = torch::where(encoder_padding_mask, NEG_INFTY, 0);
  torch::Tensor decoder_self_attention_mask = torch::where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0);
  torch::Tensor decoder_cross_attention_mask = torch::where(decoder_padding_mask_cross_attention, NEG_INFTY, 0);

  return std::make_tuple(encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask);
}

torch::Tensor scaled_dot_product(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask) {

  int d_k = q.size(-1);

  torch::Tensor scaled = torch::matmul(q, k.transpose(-1, -2)) / sqrt(d_k);

  if (!mask.defined()) {
    scaled = scaled.permute({1, 0, 2, 3}) + mask;
    scaled = scaled.permute({1, 0, 2, 3});
  }

  torch::Tensor attention = torch::softmax(scaled, -1);
  torch::Tensor out = torch::matmul(attention, v);

  return out;
}