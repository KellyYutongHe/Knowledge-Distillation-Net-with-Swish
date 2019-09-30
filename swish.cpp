#include <torch/extension.h>

#include <iostream>

#include <vector>

torch::Tensor swish(torch::Tensor z, torch::Tensor beta) {
  auto s = torch::sigmoid(beta*z);
  return z * s;
}

std::vector<at::Tensor> swish_forward(
    torch::Tensor x,
    torch::Tensor beta) {
  auto new_x = x*torch::sigmoid(beta*x);

  return {new_x};
}

std::vector<torch::Tensor> swish_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor beta) {
  auto grad_x = beta*swish(x, beta) + torch::sigmoid(beta*x)*(1-beta*swish(x, beta));
  auto grad_beta =x*x*torch::sigmoid(beta*x)*(1-torch::sigmoid(beta*x));

  return {grad_x, grad_beta};
}
