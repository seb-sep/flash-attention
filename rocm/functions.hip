#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <rocwmma/rocwmma.hpp>

#include <torch/types.h>
#include <torch/extension.h>

#include <stdio.h>
#include <iostream>
#include <utility>
#include <type_traits>

using rocwmma::row_major;
using rocwmma::col_major;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::accumulator;
using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::float64_t;



// start by making this really bad for now; keep it simple and 1 thread per vector
// not gonna have one thread per output value: a 128x128 matrix has 16384 floats in it
// assume all threads invoke this function
// adds results of A @ B to res
// no longer used in the kernel, but keep for reference
__device__
void shared_matmul(half* A, half* B, half* res, size_t m, size_t k, size_t n, bool add=true) {
    // need some scheme to evenly divide output values among input threads
    // we can be pretty naive about how to do this because we're in shared mem

    // assume this may not go in evenly or that we will have more threads than we need
    // a thread will do at least 1 unless there are less cells than threads
    size_t outputs_per_thread = max(int(m*n / blockDim.x), 1);

    // calculate flattened range and re-linearize
    for (int idx=threadIdx.x*outputs_per_thread; idx<(threadIdx.x+1)*outputs_per_thread && idx<m*n; ++idx) {
        int i = idx / n;
        int j = idx % n;

        float sum = 0;
        // so, calculate output value for i and j
        for (int q=0; q<k; ++q)
            sum += __half2float(__hmul(A[i*k + q], B[n*q + j]));
            // sum += __half2float(A[i*k + q]) * __half2float(B[n*q + j]);

        half hsum = __float2half(sum);

        // note the += here. This allows us to fuse the P @ V matmul with the addition by Oi so we don't
        // need another chunk of shared memory
        // the optionality prevents us from having to otherwise zero out a matrix first when we don't want to add
        if (add) {
            res[i*n + j] = __hadd(res[i*n + j], hsum);
        } else {
            res[i*n + j] = hsum;
        }
    }

    __syncthreads();
}


// rows and cols are the dims of the ORIGINAL matrix, not what's loaded in
// Coalesced in from, NOT to
// no syncthreads
template <typename T> 
__device__ __forceinline__ void load_coalesced_transpose(const T* from, T* to, const size_t size_from, const size_t size_to, T pad_value, size_t rows, size_t cols) {
    auto n_iters = size_to / blockDim.x + (size_to%blockDim.x != 0);
    for (int k=0; k<n_iters; ++k) {
        // indices should line up perfectly between to and from, assume 
        // the pointers are to the starts of only what you want to copy (not the potential beginning of the matrix)
        int from_idx = blockDim.x * k + threadIdx.x;
        // get the 2D indices from the linearized access, flip them to get correct transposed linearized access
        int i = from_idx / cols;
        int j = from_idx % cols;
        int to_idx = rows * j + i;
        // printf("index %d, position %d, %d in original\n", from_idx, i, j);
        // maybe still faster to have all threads loading instead of turning off enough to get an even size?
        if (from_idx < size_to)
            to[to_idx] = from_idx < size_from ? from[from_idx] : pad_value;
    }
}

// Load n_elems values from one pointer to another. Happens blockwise 1D
// if size_to less than size_from, some values not copied over
// if size_from less than size_to, padding value loaded instead
// Does NOT syncthreads
// FIXME: does it matter if inlined???
// FIXME: is it faster to call this then zero O or should the writes to shared be fused?
template <typename T> 
__device__ __forceinline__ void load_coalesced(const T* from, T* to, const size_t size_from, const size_t size_to, T pad_value) {
    auto n_iters = size_to / blockDim.x + (size_to%blockDim.x != 0);
    for (int i=0; i<n_iters; ++i) {
        // indices should line up perfectly between to and from, assume 
        // the pointers are to the starts of only what you want to copy (not the potential beginning of the matrix)
        int idx = blockDim.x * i + threadIdx.x;
        // maybe still faster to have all threads loading instead of turning off enough to get an even size?
        if (idx < size_to)
            to[idx] = idx < size_from ? from[idx] : pad_value;
    }

}