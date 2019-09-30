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
  auto grad_x = grad_y*(beta*swish(x, beta) + torch::sigmoid(beta*x)*(1-beta*swish(x, beta)));
  auto grad_beta = grad_y*(x*x*torch::sigmoid(beta*x)*(1-torch::sigmoid(beta*x)));

  return {grad_x, grad_beta};
  // return {grad_x};
}

// torch::Tensor swish(torch::Tensor z) {
//   auto s = torch::sigmoid(z);
//   return z * s;
// }
//
// std::vector<at::Tensor> swish_forward(
//     torch::Tensor x) {
//   // auto new_x = x*torch::sigmoid(x);
//   auto new_x = swish(x);
//
//   return {new_x};
// }
//
// std::vector<torch::Tensor> swish_backward(
//     torch::Tensor grad_y,
//     torch::Tensor x) {
//   auto grad_x = grad_y*(swish(x) + torch::sigmoid(x)*(1-swish(x)));
//
//   // return {grad_x, grad_beta};
//   return {grad_x};
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &swish_forward, "SWISH forward");
  m.def("backward", &swish_backward, "SWISH backward");
}
