#include <cuda.h>
#include <torch/types.h>
#include <stdio.h>
#include <utility>

const size_t TILE_SIZE = 4;

// Calculate blockwise normalization constant for input matrix
// Each thread block will write a single value into max_result and sum_result, 
__global__
void online_softmax_norm_constant_kernel(const float* max_inputs, const float* sum_inputs, float* max_results, float* sum_results, size_t size) {
    // first, copy into shared
    // theoretically, this should coalesce acceses as long as each thread accesses in the same position
    // two tiles: one for the running max for numerically safe softmax, 
    // another for the sum of exps to calculate normalization constant
    __shared__ float maxes[TILE_SIZE];
    __shared__ float sums[TILE_SIZE];

    uint idx = blockIdx.x*blockDim.x + threadIdx.x;

    // save original matrix value in private register for calculating final result later
    // FIXME: this wastes registers on some threads, maybe fix later
    // this may not be necessary anymore, would go in elementwise kernel
    float original_value;

    if (idx < size) {
        // load in maxes (just the value on first iteration)
        maxes[threadIdx.x] = max_inputs[idx];
        
        // populate sums with 1, the base case on the first iteration and the actual value on others
        sums[threadIdx.x] = (sum_inputs == nullptr) ? 1.0f : sum_inputs[idx];
    } else {
        maxes[threadIdx.x] = -FLT_MAX; // pad with -inf, because max with -inf is the identity
        sums[threadIdx.x] = 0.0f;
    }

    // printf("Loaded in max of %f and sum of %f\n", maxes[threadIdx.x], sums[threadIdx.x]);

    // loop for how many reductions we need to do
    // now, instead of strides increasing, they decrease
    // start halfway across, then decrease
    // NOTE: half the threads in the kernel stop doing work here. You could instead
    // have each thread load in two values and you can now launch twice the # of threads 
    // in this case, reduces accumulate in 0 index because last loop is stride of 1
    uint i = threadIdx.x;
    for (int stride=TILE_SIZE/2; stride>=1; stride/=2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            uint j = threadIdx.x + stride;
            float new_max = max(maxes[i], maxes[j]);
            sums[i] = sums[i] * __expf(maxes[i] - new_max) + sums[j] * __expf(maxes[j] - new_max);
            maxes[i] = new_max;
        }
    }
    
    // we don't need another syncthreads here, because on the last iteration of the loop,
    // only thread 0 is active
    // size of results vectors is implicit based on TILE_SIZE and input size
    if (threadIdx.x == 0) {
        max_results[blockIdx.x] = maxes[0];
        sum_results[blockIdx.x] = sums[0];
    }
}

__global__
void softmax_elementwise_kernel(const float* matrix, float* result, float exp_sum, float max_value, size_t size) {
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;

    // because we only do one read and one write per value, don't worry abt shared
    if (idx < size) {
        float dv_inv = __fdividef(1.0f, exp_sum);
        result[idx] = __expf(matrix[idx] - max_value) * dv_inv;
    }
}

/*
Strategy:
norm constant kernel and elementwise softmax kernel
loop norm constant kernel until we can reduce in a single thread block
*/
torch::Tensor softmax(torch::Tensor matrix) {

    // length of input vectors
    // logically, this should never be 0 because the length of input is prev output
    int input_size = matrix.numel();
    // length of output vectors
    // this relationship should always match
    int output_size = input_size / TILE_SIZE + (input_size%TILE_SIZE != 0);

    torch::Tensor results = torch::empty_like(matrix);

    // keep looping the norm constant kernel until inputs fit in a single thread block
    // the last kernel call should be when it fits in a single tile
    auto options = torch::TensorOptions().device(matrix.device()).dtype(matrix.dtype()); 
    torch::Tensor max_inputs = results; // can reuse result matrix for one of these intermediaries
    torch::Tensor sum_inputs = torch::empty({input_size}, options);
    torch::Tensor max_results = torch::empty({input_size}, options);
    torch::Tensor sum_results = torch::empty({input_size}, options);

    // favor keeping this separate, because it's invoked in a different way than the others
    online_softmax_norm_constant_kernel<<<output_size, TILE_SIZE>>>(
        (float*)matrix.const_data_ptr(), nullptr,
        (float*)max_inputs.mutable_data_ptr(), (float*)sum_inputs.mutable_data_ptr(), input_size);

    input_size = output_size;
    while (input_size > 1) {
        output_size = input_size / TILE_SIZE + (input_size%TILE_SIZE != 0);
        online_softmax_norm_constant_kernel<<<output_size, TILE_SIZE>>>(
            (float*)max_inputs.const_data_ptr(), (float*)sum_inputs.mutable_data_ptr(), 
            (float*)max_results.const_data_ptr(), (float*)sum_results.mutable_data_ptr(), input_size);

        // swap inputs and outputs for next call
        // this means that after all iterations are said and done, final results will be in INPUT tensors
        std::swap(max_inputs, max_results);
        std::swap(sum_inputs, sum_results);
        input_size = output_size;
    }

    float exp_sum = sum_inputs[0].item<float>();
    float max_value = max_inputs[0].item<float>();
    
    // at this point, we assume we only launch a single thread block
    softmax_elementwise_kernel<<<1, TILE_SIZE>>>(
        (float*)matrix.const_data_ptr(), (float*)results.mutable_data_ptr(), exp_sum, max_value, matrix.numel());

    return results;
}
 