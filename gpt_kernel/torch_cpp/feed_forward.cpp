#include "feed_forward.hpp"

FeedForwardImpl::FeedForwardImpl(int64_t d_model, int64_t hidden, double drop_prob)
    : net(register_module("net", torch::nn::Sequential(
          torch::nn::Linear(d_model, hidden),
          torch::nn::ReLU(),
          torch::nn::Linear(hidden, d_model),
          torch::nn::Dropout(drop_prob)))) {}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {

    return net->forward(x);
}