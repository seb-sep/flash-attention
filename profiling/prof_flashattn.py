import torch
import torch.nn.functional as F

from torch.utils.cpp_extension import _import_module_from_library

module = _import_module_from_library('m_v23', './build', True)

# instantiate tensors
# dims = ((48, 32), (64, 256), (1024, 256), (4096, 1024))
N, d = 4096, 1024
n_iters = 1000
module.run_bench(N, d, n_iters)
    # F.scaled_dot_product_attention(Q, K, V, scale=1)