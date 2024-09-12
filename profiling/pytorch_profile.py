import torch
from torch.utils.cpp_extension import _import_module_from_library
import torch.utils.benchmark as benchmark
from torch.nn.functional import scaled_dot_product_attention

# Instantiate custon and naive flash attn
print("Loading custom FlashAttention:")
my_flashattn = _import_module_from_library('m', './build', True).flash_attn

pytorch_attn = scaled_dot_product_attention

print("torch.compiling naive attention:")
@torch.compile
def naive_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """equivalent to F.scaled_dot_product_attention(Q, K, V, scale=1)"""
    return torch.softmax((Q)@ K.T, dim=1) @ V
 
def run_flash_attention_comparison(N, d, num_iterations):
    # Setup inputs
    Q = torch.rand((N, d), dtype=torch.float16).cuda()
    K = torch.rand((N, d), dtype=torch.float16).cuda()
    V = torch.rand((N, d), dtype=torch.float16).cuda()

    # Warmup
    print("Warming up...")
    for _ in range(10):
        my_flashattn(Q, K, V)
        naive_attn(Q, K, V)

    custom_timer = benchmark.Timer(
        stmt='my_flashattn(Q, K, V)',
        setup='from __main__ import my_flashattn',
        globals={'Q': Q, 'K': K, 'V': V}
    )
    
    naive_timer = benchmark.Timer(
        stmt='naive_attn(Q, K, V)',
        setup='from __main__ import naive_attn',
        globals={'Q': Q, 'K': K, 'V': V}
    )

    pytorch_timer = benchmark.Timer(
        stmt='pytorch_attn(Q, K, V, scale=1)',
        setup='from __main__ import pytorch_attn',
        globals={'Q': Q, 'K': K, 'V': V}
    )
    # return F.scaled_dot_product_attention(Q, K, V, scale=1)

    print("Timing for custom FlashAttention:")
    print(custom_timer.timeit(num_iterations))
    print("Timing for naive torch.compiled attention:")
    print(naive_timer.timeit(num_iterations))
    print("Timing for builtin PyTorch attention:")
    print(pytorch_timer.timeit(num_iterations))

N, d = 16384, 128
num_iterations = 10
run_flash_attention_comparison(N, d, num_iterations)