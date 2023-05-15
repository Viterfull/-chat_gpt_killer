#pragma once

#include <torch/torch.h>

torch::Device get_device();

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_masks(
    torch::Tensor eng_batch, torch::Tensor kn_batch);

torch::Tensor scaled_dot_product(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask = torch::Tensor());
