#include <torch/extension.h>

torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attn", torch::wrap_pybind_function(flash_attention), "flasn_attn");
}