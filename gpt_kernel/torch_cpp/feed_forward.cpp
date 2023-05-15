#include "feed_forward.hpp"

FeedFowardImpl::FeedFowardImpl(int64_t d_model, int64_t hidden, double drop_prob)
    : net(register_module("net", torch::nn::Sequential(
          torch::nn::Linear(d_model, hidden),
          torch::nn::ReLU(),
          torch::nn::Linear(hidden, d_model),
          torch::nn::Dropout(drop_prob)))) {}

torch::Tensor FeedFowardImpl::forward(torch::Tensor x) {

    return net->forward(x);
}