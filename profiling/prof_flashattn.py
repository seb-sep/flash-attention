import torch
import torch.nn.functional as F

# from torch.utils.cpp_extension import load
# module = load(
#     name='m',
#     # sources=['cuda/main.cpp', 'cuda/flash_attention.cu'],
#     sources=['rocm/flash_attention.hip',],
#     extra_cflags=['--offload-arch="gfx1100"',],
#     verbose=True
# )

# print('module build successfully')

# instantiate tensors
# Q = torch.rand((8, 2), dtype=torch.float32, device='cuda')
# K = torch.rand((8, 2), dtype=torch.float32, device='cuda')
# V = torch.rand((8, 2), dtype=torch.float32, device='cuda')

Q = torch.randint(1, 9, (4, 2), device='cuda').float()
K = torch.randint(1, 9, (4, 2), device='cuda').float()
V = torch.randint(1, 9, (4, 2), device='cuda').float()

for _ in range(1):
    # module.flash_attn(Q, K, V)
    F.scaled_dot_product_attention(Q, K, V, scale=1)