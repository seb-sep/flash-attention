
#include <cuda.h>
#include <torch/types.h>
#include <stdio.h>
#include <iostream>
#include <utility>

// in flash attn 2, set to either 64 or 128, can basically pick what you want as long as 
// you have the sram for it
// also number of threads per block
// wait HUGE heuristic: flash attn 2 typically uses 4 or 8 warps per thread block
// sizes of QKV tiles are {64, 128} x {64, 128}
// Is this referring to Br and Bc? Maybe this is kBlockM, N in the cuda code
// https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h
// FIXME: currently doesn't work when too many threads for the matmuls, figure out better allocation of threads
const size_t N_WARPS = 1;
const size_t Bc = 4;
const size_t Br = 4;

__device__
void tensorprint(const float* data, const size_t rows, const size_t cols) {
    printf("[");
    for (int i=0; i<rows; ++i) {
        printf("[");
        for (int j=0; j<cols; ++j) {
            printf("%.4f, ", data[i*cols + j]);
        }
        printf("]");
        if (i<rows-1)
            printf("\n");
    }
    printf("]\n\n");
}

// start by making this really bad for now; keep it simple and 1 thread per vector
// not gonna have one thread per output value: a 128x128 matrix has 16384 floats in it
// assume all threads invoke this function
// adds results of A @ B to res
__device__
void shared_matmul(const float* A, const float* B, float* res, size_t m, size_t k, size_t n, bool add=true) {
    // need some scheme to evenly divide output values among input threads
    // we can be pretty naive about how to do this because we're in shared mem

    // this should go in evenly
    size_t outputs_per_thread = m*n / blockDim.x;
    // calculate flattened range and re-linearize
    for (int idx=threadIdx.x*outputs_per_thread; idx<(threadIdx.x+1)*outputs_per_thread; ++idx) {
        int i = idx / n;
        int j = idx % n;

        float sum = 0;
        // so, calculate output value for i and j
        for (int q=0; q<k; ++q)
            sum += A[i*k + q] * B[n*q + j];
        
        // note the += here. This allows us to fuse the P @ V matmul with the addition by Oi so we don't
        // need another chunk of shared memory
        // the optionality prevents us from having to otherwise zero out a matrix first
        if (add) {
            res[i*n + j] += sum;
        } else {
            res[i*n + j] = sum;
        }
    }

    __syncthreads();
}


__global__
void flash_attention_kernel(const float *Q, const float* K, const float* V, float* O, float* m, float* l, size_t N, size_t d, size_t tc) {


    /*
    What work is done by a single thread block and a single thread?
    - Thread block calculates one tile of QKV at a time
    - Thread block given a whole row
    - Therefore, thread blocks organized by tiles of Q
    - So, I have have a 1D row of thread blocks
    - Dim of thread block ideally matches dim of tile?
    - Don't need to worry about number of thread blocks, but remember each thread block can only have <=1024 threads
    - easy option is one thread per vector in a tile (so bc/br)
    - How to get a Bc ~= M/4d multiple of 32, in a way ideally irrespective of d?
    32 = M/4d
    128d = M
    So as long as your M is some 128cd, where c is an int, good to go
    M can be bigger than 1024, but num threads cannot
    - https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
    - As it turns out, in flash attn 2 and 3 you can pick whatever bc and br you want
    - Then, you go back and allocate your shared memory to be whatever you need
    */ 

    // index in the range of tr
    size_t j = blockIdx.x;

    // now that Bc and Br are statically known, can do statically allocated sram
    // dynamically allocate shared memory, because we don't know 
    // what our tile sizes will be until runtime (d is dynamic)
    extern __shared__ float sram[];

    // // total size of shared k and v is num of vecs in tile * size of vec
    size_t kv_size = Bc * d;
    float *K_s = sram;
    float *V_s = sram + kv_size;

    size_t qo_size = Br * d;
    float *Q_s = V_s + kv_size;
    float *O_s = Q_s + qo_size;

    // {Brxd} @ {dxBc} = {BrxBc}
    size_t p_size = Br * Bc;
    float *P_s = O_s + qo_size;

    float* m_s = P_s + p_size;
    // should be br, because we need Br values of m and l in each shared block
    float* l_s = m_s + Br;



    // parallelizing over queries, so load in Q, O tile first
    // remember that each thread block should load in same # of values (except for last)
    // probably assume we have more threads than vectors (everything likely always mult of 32 tho)
    // in this case, we only care about mapping thread index to Bc
    // with the hyperparam possibilities laid out in flash attn 2, alwyas twice the num of threads
    // remember that your tiles are 2D; each thread loads in a whole vector
    int r = blockDim.x / Br;
    if (int idx = threadIdx.x/r; threadIdx.x % r == 0 && j*Br+idx< N) {
        for (int k=0; k<d; ++k) {
            Q_s[idx*d + k] = Q[(j*Br + idx)*d + k];
            O_s[idx*d + k] = O[(j*Br + idx)*d + k];
        }
        // only load a single value for m and l vector tiles
        m_s[idx] = m[j*Br + idx];
        l_s[idx] = l[j*Br + idx];
    }
    __syncthreads();

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     tensorprint(Q_s, Br, d);
    //     tensorprint(O_s, Br, d);
    //     tensorprint(m_s, 1, Br);
    //     tensorprint(l_s, 1, Br);
    // }

    r = blockDim.x / Bc;
    // loop over kv tiles
    for (size_t i=0; i<tc; ++i) {
        // load tile of k and v into sram
        if (int idx = threadIdx.x/r; threadIdx.x % r == 0 && i*Bc+idx< N) {
            for (int k=0; k<d; ++k) {
                // need to load K transposed; load column into K_s
                // K_s has rows of len Bc and cols of len d
                K_s[Bc*k + idx] = K[(i*Bc + idx)*d + k];
                V_s[idx*d + k] = V[(i*Bc + idx)*d + k];
            }
        }
        __syncthreads();

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     tensorprint(K_s, d, Bc);
        //     tensorprint(V_s, Bc, d);
        // }

        // matmul QK^T
        shared_matmul(Q_s, K_s, P_s, Br, d, Bc, false);
    

        // define sumexp l_new to divide out at the end
        // no need for shared memory, because row-wise
        float l_new;
        // calculate new maxes
        // for all these row-wise ops, just use the first Br threads per block to minimize warp divergence
        if (threadIdx.x < Br) {

            // calculate max over a single row of P_s
            float mt = -INFINITY;
            for (int i=0; i<Bc; ++i)
                mt = max(mt, P_s[threadIdx.x*Bc + i]);
            
            // shifted exponentiation of each row and row-wise sumexp
            float lt = 0;
            for (int i=0; i<Bc; ++i) {
                float exp = __expf(P_s[threadIdx.x*Bc + i] - mt);
                P_s[threadIdx.x*Bc + i] = exp;
                lt += exp;
            }
            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("P_s after shifted exp:\n");
            //     tensorprint(P_s, Br, Bc);
            // }

            // compute new m and l for your row
            float m_new = max(mt, m_s[threadIdx.x]);
            float old_lse = __expf(m_s[threadIdx.x] - m_new) * l_s[threadIdx.x];
            l_new = old_lse + __expf(mt - m_new) * lt;
            
            // on this first iteration, O_s should be all 0's
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("O_s should be zeros on first iter, than something else (before being multiplied by %f):\n", old_lse);
                tensorprint(O_s, Br, d);
            }

            // row-wise multiply of old output values
            for (int i=0; i<d; ++i)
                O_s[threadIdx.x*d + i] *= old_lse;

            
            
            // scale P's rows by exp(mt - m_new) as in flashattn
            for (int i=0; i<Bc; ++i)
                P_s[threadIdx.x*Bc+i] *= __expf(mt - m_new);
            
            // printf("multiplied by %f\n", __expf(mt - m_new));
            // load softmax statistics back to shared memory as the 'old' values
            // FIXME: what of this needs to be in shared memory and what can just go in registers?
            m_s[threadIdx.x] = m_new;
            l_s[threadIdx.x] = l_new;


        }
        __syncthreads();
            // should be the same value after exponentiation on the first iteration
            // if (threadidx.x == 0 && blockidx.x == 0) {
            //     printf("p_s should be the same as before\n");
            //     tensorprint(P_s, Br, Bc);
            //     printf("V_s is:\n");
            //     tensorprint(V_s, Bc, d);

            // }

        shared_matmul(P_s, V_s, O_s, Br, Bc, d);

        // divide new O by l_new
        if (threadIdx.x < Br) {
            float dividend = __fdividef(1, l_new);
            for (int i=0; i<d; ++i)
                O_s[threadIdx.x*d + i] *= dividend;
        }
            
        if (threadIdx.x == 0 && blockIdx.x == 0)
            tensorprint(O_s, Br, d);


        // don't need to write back to hbm as in the pytorch version, because a thread 
        // block iterates only over a single row of KV instead of the whole attn matrix
    }

    // write tile of O back to global results
    r = blockDim.x / Br;
    if (int idx = threadIdx.x/r; threadIdx.x % r == 0 && j*Br+idx< N) {
        for (int k=0; k<d; ++k) {
             O[(j*Br + idx)*d + k] = O_s[idx*d + k];
        }
        // don't worry about saving softmax statistics for now

    }
    __syncthreads();

}


torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {

    // Q, K, V should be same size
    assert(Q.sizes() == K.sizes() && K.sizes() == V.sizes());
    
    size_t N = Q.size(0);
    size_t d = Q.size(1);

    // thread block size dictates how many vectors per QKV block, and how many thread blocks required

    // // hold up all these values are knowable before runtime
    // // size and num of KV tiles
    size_t tc = ceil(N / Bc);

    // // size and num of Q, O tiles
    size_t tr = ceil(N / Br);
    std::cout << tr << tc << '\n';

    // instantiate extra tensors
    auto options = torch::TensorOptions().device(Q.device()).dtype(Q.dtype()); 
    torch::Tensor O = torch::zeros_like(Q);
    torch::Tensor m = torch::full({int(N),}, -INFINITY, options);
    torch::Tensor l = torch::zeros_like(m);

    // each thread block loads a SINGLE Q tile, iterates over all respective KV, so it only cares
    // abt size of Q tile, so use tr for num thread blocks
    // how much sram to allocate?
    size_t kv_size = Bc * d;
    size_t qo_size = Br * d;
    size_t p_size = Br * Bc;
    // shared for K, V, Q, O, m, l
    size_t smem_size = sizeof(float) * (2*kv_size + 2*qo_size + p_size + 2*Br);

    // want flat thread grid
    flash_attention_kernel<<<tr, N_WARPS*32, smem_size>>>(
        (float*)Q.const_data_ptr(), (float*)K.const_data_ptr(), (float*)V.const_data_ptr(), 
        (float*)O.mutable_data_ptr(), (float*)m.mutable_data_ptr(), (float*)l.mutable_data_ptr(),
        N, d, tc);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return torch::Tensor();
    }

    return O;
}