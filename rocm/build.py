from torch.utils.cpp_extension import load
import os

print("Current working directory:", os.getcwd())

module = load(
    name='m',
    sources=['rocm/flash_attention.hip',],
    extra_cflags=['--offload-arch="gfx1100"'],
    build_directory='build',
    verbose=True
)