#include <torch/extension.h>

torch::Tensor softmax(torch::Tensor matrix);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax", torch::wrap_pybind_function(softmax), "softmax");
}