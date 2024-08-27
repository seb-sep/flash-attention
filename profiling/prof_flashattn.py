import torch
import torch.nn.functional as F

from torch.utils.cpp_extension import _import_module_from_library

module = _import_module_from_library('m_v1', './build', True)

# instantiate tensors
# Q = torch.rand((8, 2), dtype=torch.float32, device='cuda')
# K = torch.rand((8, 2), dtype=torch.float32, device='cuda')
# V = torch.rand((8, 2), dtype=torch.float32, device='cuda')

Q = torch.randint(1, 9, (4, 2), device='cuda').float()
K = torch.randint(1, 9, (4, 2), device='cuda').float()
V = torch.randint(1, 9, (4, 2), device='cuda').float()

for _ in range(100):
    module.flash_attn(Q, K, V)
    # F.scaled_dot_product_attention(Q, K, V, scale=1)