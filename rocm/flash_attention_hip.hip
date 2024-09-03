#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <torch/types.h>
#include <torch/extension.h>

#include <stdio.h>
#include <iostream>
#include <utility>

/*
BIG ISSUE: when the input Q has a O, and K has some -infs (because of padding), output is nan instead of -inf, which propagates down
*/

// in flash attn 2, set to either 64 or 128, can basically pick what you want as long as 
// you have the sram for it
// also number of threads per block
// wait HUGE heuristic: flash attn 2 typically uses 4 or 8 warps per thread block
// sizes of QKV tiles are {64, 128} x {64, 128}
// Is this referring to Br and Bc? Maybe this is kBlockM, N in the cuda code
// https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h
// FIXME: currently doesn't work when too many threads for the matmuls, figure out better allocation of threads
// because of sram limitations, I can't afford the same block sizes and thus I should probably allocate less warps per block
// Br should always be at least 32, since many parts of the kernel are row-wise over tiles of height Br
// Furthermore, as long as things are threadwise over Br, Br should always be larger than Bc and generally as large as possible
const size_t Br = 32;
const size_t Bc = 48;
// const size_t Br = 8;
// const size_t Bc = 4;
const size_t N_WARPS = Br / 32 + (Br%32!=0);


__device__
void tensorprint(half* data, const size_t rows, const size_t cols) {
    printf("[");
    for (int i=0; i<rows; ++i) {
        printf("[");
        for (int j=0; j<cols; ++j) {
            printf("%.4f, ", __half2float(data[i*cols + j]));
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
// FIXME: do your m and l vectors need to be mixed precision? also revisit whether you need them in global
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

__device__ __forceinline__ 
half max(half a, half b) { return __hgt(a, b) ? a : b; }

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


__global__
void flash_attention_kernel(const half *Q, const half* K, const half* V, half* O, float scale, size_t N, size_t d, size_t tc) {


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
        
    const half H_ZERO = __float2half(0);
    const half2 H2_ZERO = __float2half2_rn(0);
    // use largest negative instead of inf, because if you ever multiply -inf by 0 you're in big trouble
    const half H_MIN = __float2half(-65500.0f);
    const half2 H2_MIN = __float2half2_rn(-65500.0f);
    
    // now that Bc and Br are statically known, can do statically allocated sram
    // dynamically allocate shared memory, because we don't know 
    // what our tile sizes will be until runtime (d is dynamic)
    extern __shared__ half sram[];

    // // total size of shared k and v is num of vecs in tile * size of vec
    size_t kv_size = Bc * d;
    half *K_s = sram;
    half *V_s = sram + kv_size;

    size_t qo_size = Br * d;
    half *Q_s = V_s + kv_size;
    half *O_s = Q_s + qo_size;

    // {Brxd} @ {dxBc} = {BrxBc}
    size_t p_size = Br * Bc;
    half *P_s = O_s + qo_size;




    // half* m_s = P_s + p_size;
    // // should be br, because we need Br values of m and l in each shared block
    // half* l_s = m_s + Br;

    // try swapping m_s and l_s for registers, since each thread currently only uses one apiece
    float m_s = -INFINITY;
    float l_s = 1;

    // parallelizing over queries, so load in Q, O tile first
    // remember that each thread block should load in same # of values (except for last)
    // probably assume we have more threads than vectors (everything likely always mult of 32 tho)
    // in this case, we only care about mapping thread index to Bc
    // remember that your tiles are 2D; each thread loads in a whole vector

    // Index of the Q/O vector to start from
    int qo_idx = Br * blockIdx.x;
    // if (threadIdx.x < Br) {
    //     // index for a VECTOR of Q
    //     if (idx < N) {
    //         for (int k=0; k<d; ++k) {
    //             // apply scaling factor here instead of after QK^T; this prevents matmul from overflowing
    //             Q_s[threadIdx.x*d + k] = Q[idx*d + k];
    //             O_s[threadIdx.x*d + k] = H_ZERO; //O[(j*Br + idx)*d + k];
    //         }
    //     } else {
    //         for (int k=0; k<d; ++k) {
    //             Q_s[threadIdx.x*d + k] = H_NINF;
    //             O_s[threadIdx.x*d + k] = H_ZERO;
    //         }
    //     }
    // }
    // load from should be however many vals left; if this happens to be >=N, nothing changes
    // otherwise, it loads the padded value
    // load_coalesced(&Q[qo_idx*d], Q_s, (N-qo_idx)*d, Br*d, H_ZERO);
    // load_coalesced((half*)nullptr, O_s, 0, Br*d, H_ZERO); // zero out shared
    // try inlining and fusing coalesced loads
    size_t size_from = min(Br, N-qo_idx)*d;
    size_t size_to = Br*d;
    // assume d always divisible by 2
    for (int i=0; i<d/2; ++i) {
        // indices should line up perfectly between to and from, assume 
        // the pointers are to the starts of only what you want to copy (not the potential beginning of the matrix)
        // thread indices are doubled, since each loads in two at a time
        int idx = blockDim.x * i + threadIdx.x;
        // maybe still faster to have all threads loading instead of turning off enough to get an even size?
        // we know that we will never exceed the Br width here
        // when we can load in both at a time
        // there will only be this bad warp divergence in a single warp, so not too bad
        if (idx*2+1 < size_from) {
            reinterpret_cast<half2*>(Q_s)[idx] = reinterpret_cast<const half2*>(&Q[qo_idx*d])[idx];
        } else if (idx*2 < size_from) {
            // when using idx in the half2 vector, the pointer arithmetic is DOUBLED, so double here for parity
            // I doubt that vectorizing this will provide much speedup, since shared is already pretty fast
            Q_s[idx*2] = Q[qo_idx*d + idx*2];
            Q_s[idx*2+1] = H_ZERO;
        } else {
            // printf("Loading zeros into shared indices %d and %d\n", idx*2, idx*2+1);
            reinterpret_cast<half2*>(Q_s)[idx] = H2_ZERO;
        }
        reinterpret_cast<half2*>(O_s)[idx] = H2_ZERO;
    }
    // save the syncthreads for later

    // __syncthreads();
    // if (threadIdx.x == 0 && blockIdx.x == 1) {
    //     printf("Q block %d:\n", blockIdx.x);
    //     tensorprint(Q_s, Br, d);
    //     printf("O:\n");
    //     tensorprint(O_s, Br, d);
    // }

    int r = blockDim.x / Bc;
    // loop over kv tiles
    int kv_idx;
    // load vectorized to these regs, then place the values in shared
    half k_temp[2];
    for (size_t i=0; i<tc; ++i) {

        // Index of the KV vector to start from
        kv_idx = Bc * i;

        // load tile of k and v into sram
        // load_coalesced_transpose(&K[kv_idx*d], K_s, (N-kv_idx)*d, Bc*d, H_MIN, Bc, d);
        // load_coalesced(&V[kv_idx*d], V_s, (N-kv_idx)*d, Bc*d, H_ZERO);
        size_to = Bc*d;
        size_from = min(Bc, N-kv_idx)*d;
        int n_iters = size_to / (blockDim.x*2) + (size_to%(blockDim.x*2) != 0);
        // can't vectorize loads of K, since you need to place it transposed in shared...
        // UNLESS you load vectorized to a register and then load to shared
        for (int k=0; k<n_iters; ++k) {
            // indices should line up perfectly between to and from, assume 
            // the pointers are to the starts of only what you want to copy (not the potential beginning of the matrix)
            // remember that the DOUBLED index into the matrix
            int idx = blockDim.x * k + threadIdx.x;
            // calculate indices for transposed load
            int i = idx*2 / d;
            int j = idx*2 % d;
            int to_idx = Bc * j + i;
            
            if (idx*2+1 < size_from) {
                reinterpret_cast<half2*>(V_s)[idx] = reinterpret_cast<const half2*>(&V[kv_idx*d])[idx];
                
                reinterpret_cast<half2*>(k_temp)[0] = reinterpret_cast<const half2*>(&K[kv_idx*d])[idx];
                // printf("Loading %f %f to (%d %d), (%d %d)\n", __half2float(k_temp[0]), __half2float(k_temp[1]), j, i, j+1, i);
                K_s[to_idx] = k_temp[0];
                K_s[to_idx + Bc] = k_temp[1];
            } else if (idx*2 < size_from) {
                V_s[idx*2] = V[kv_idx*d + idx*2];
                V_s[idx*2+1] = H_ZERO;

                // printf("Loading %f zero to (%d %d), (%d %d) (linearized %d, %d)\n", __half2float(K[kv_idx*d + idx*2]), j, i, j+1, i, to_idx, to_idx+Bc);
                K_s[to_idx] = K[kv_idx*d + idx*2];
                K_s[to_idx + Bc] = H_MIN;
            } else if (idx*2 < size_to) {
                reinterpret_cast<half2*>(V_s)[idx] = H2_ZERO;
                
                // printf("Loading zeros to (%d %d), (%d %d) (linearized %d, %d)\n", j, i, j+1, i, to_idx, to_idx + Bc);
                K_s[to_idx] = H_MIN;
                K_s[to_idx + Bc] = H_MIN;
            }
        }
        
        __syncthreads();

        // if (threadIdx.x == 0 && blockIdx.x==0) {
        //     printf("K^T block %d:\n", blockIdx.x);
        //     tensorprint(K_s, d, Bc);
        //     printf("V block %d:\n", blockIdx.x);
        //     tensorprint(V_s, Bc, d);
        // }

        // // matmul QK^T
        // if (threadIdx.x == 0 && blockIdx.x == 0)
        //     printf("Performkng QKT");
        shared_matmul(Q_s, K_s, P_s, Br, d, Bc, false);
        // if (threadIdx.x == 0 && blockIdx.x == 2) {
        //         printf("P_s before shifted exp:\n");
        //         tensorprint(P_s, Br, Bc);
        // }

        // define sumexp l_new to divide out at the end
        // no need for shared memory, because row-wise
        float l_new;
        // calculate new maxes
        // for all these row-wise ops, just use the first Br threads per block to minimize warp divergence

        // calculate max over a single row of P_s
        half mt_h = H_MIN;
        for (int i=0; i<Bc; ++i)
            mt_h = max(mt_h, P_s[threadIdx.x*Bc + i]);
        float mt = __half2float(mt_h);
        // shifted exponentiation of each row and row-wise sumexp
        float lt = 0;
        for (int i=0; i<Bc; ++i) {
            float exp = __expf(__half2float(P_s[threadIdx.x*Bc + i]) - mt);
            P_s[threadIdx.x*Bc + i] = __float2half(exp);
            lt += exp;
        }
        // if (threadIdx.x == 0 && blockIdx.x == 2) {
        //     printf("P_s after shifted exp:\n");
        //     tensorprint(P_s, Br, Bc);
        // }

        // compute new m and l for your row
        float m_new = max(mt, m_s);
        float old_lse = __expf(m_s - m_new) * l_s;
        float undo_exp = __expf(mt - m_new);
        l_new = old_lse + undo_exp * lt;
        // printf("%d: undo exp is %f, from an mt of %f\n", threadIdx.x, undo_exp, mt);
        
        // on this first iteration, O_s should be all 0's
        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     printf("O_s should be zeros on first iter, than something else (before being multiplied by %f):\n", old_lse);
        //     tensorprint(O_s, Br, d);
        // }

        // row-wise multiply of old output values
        // half old_lse_h = __float2half(old_lse);

        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     printf("Old Oi before getting renomalized:\n");
        //     tensorprint(O_s, Br, d);
        // }
        for (int i=0; i<d; ++i)
            O_s[threadIdx.x*d + i] = __hmul(O_s[threadIdx.x*d + i], __float2half(old_lse));
        
        // only necessary for now
        // __syncthreads();

        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     printf("Oi after getting renomalized:\n");
        //     tensorprint(O_s, Br, d);
        // }
        // scale P's rows by exp(mt - m_new) as in flashattn
        // half undo_exp_h = __float2half(undo_exp);
        for (int i=0; i<Bc; ++i)
            P_s[threadIdx.x*Bc+i] = __hmul(P_s[threadIdx.x*Bc+i], __float2half(undo_exp));
        
        // printf("multiplied by %f\n", __expf(mt - m_new));
        // load softmax statistics back to shared memory as the 'old' values
        // FIXME: what of this needs to be in shared memory and what can just go in registers?
        m_s = m_new;
        l_s = l_new;


        __syncthreads();
        // should be the same value after exponentiation on the first iteration
        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     printf("p_s should be the same as before\n");
        //     tensorprint(P_s, Br, Bc);
        //     printf("V_s is:\n");
        //     tensorprint(V_s, Bc, d);

        // }

        // if (threadIdx.x == 0 && blockIdx.x == 0)
        //     printf("Performkng PV");
        shared_matmul(P_s, V_s, O_s, Br, Bc, d);
        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     printf("Oi after the matmul:\n");
        //     tensorprint(O_s, Br, d);
        // }

        // divide new O by l_new
        half dividend = __float2half(__fdividef(1, l_new));
        // if (__hisinf(dividend) || __hisnan(dividend))
        //     printf("dividend is inf or nan, l_new is %f\n", l_new);
        // printf("l_new is %f, dividend is %f\n", l_new, dividend);
        for (int i=0; i<d; ++i)
            O_s[threadIdx.x*d + i] = __hmul(O_s[threadIdx.x*d + i], dividend);
            
        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     printf("Oi after divided by l_new; end of block:\n");
        //     tensorprint(O_s, Br, d);
        // }


        // don't need to write back to hbm as in the pytorch version, because a thread 
        // block iterates only over a single row of KV instead of the whole attn matrix
    }

    // write tile of O back to global results
    // a bunch of these will be garbage if from padded values
    // if (threadIdx.x < Br && idx < N) {
    //     for (int k=0; k<d; ++k) {
    //         // printf("Writing %f back to dimension %d of output token %d\n", O_s[idx*d + k], k, idx);
    //         O[idx*d + k] = O_s[threadIdx.x*d + k];
    //     }
    //     // don't worry about saving softmax statistics for now

    // }
    // qo_idx + Br < N, then Br, otherwise, N - qo-idx
    // load_coalesced(O_s, &O[qo_idx*d], Br*d, min(Br, N-qo_idx)*d, H_ZERO);
    size_to = min(Br, N-qo_idx)*d;
    size_from = Br*d;
    for (int i=0; i<d/2; ++i) {
        // indices should line up perfectly between to and from, assume 
        // the pointers are to the starts of only what you want to copy (not the potential beginning of the matrix)
        int idx = blockDim.x * i + threadIdx.x;
        // maybe still faster to have all threads loading instead of turning off enough to get an even size?
        // we know that we will never exceed the Br width here
        if (idx*2+1 < size_to) {
            reinterpret_cast<half2*>(&O[qo_idx*d])[idx] = reinterpret_cast<half2*>(O_s)[idx];
        } else if (idx*2 < size_to) {
            O[qo_idx*d + idx*2] = O_s[idx*2];
        }
    }

   
    __syncthreads();

}


torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);

    std::cout << "  Shared Memory per Block: " << props.sharedMemPerBlock << std::endl;
    std::cout << "  Shared Memory Banks: " << props.regsPerBlock << std::endl;
    std::cout << "  Warp Size: " << props.warpSize << std::endl;
    std::cout << "  Max Threads per Block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "  Constant Memory: " << props.totalConstMem << std::endl;
    std::cout << "  L2 Cache: " << props.l2CacheSize << std::endl;
    std::cout << std::endl;


    if (!(Q.sizes() == K.sizes() && K.sizes() == V.sizes())) {
       printf("Q, K, V must be same size\n");
       return Q;
    } 

    if (!(Q.dtype() == torch::kHalf && K.dtype() == torch::kHalf && V.dtype() == torch::kHalf)) {
        printf("kernel implemented in fp16 only\n");
        return Q;
    } 

    
    size_t N = Q.size(0);
    size_t d = Q.size(1);

    // thread block size dictates how many vectors per QKV block, and how many thread blocks required

    // // hold up all these values are knowable before runtime
    // // size and num of KV tiles
    size_t tc = N / Bc + (N%Bc!=0);

    // // size and num of Q, O tiles
    size_t tr = N / Br + (N%Br!=0);
    printf("tc=%d, tr=%d\n", tc, tr);

    // instantiate extra tensors
    auto options = torch::TensorOptions().device(Q.device()).dtype(Q.dtype()); 
    torch::Tensor O = torch::empty_like(Q);

    // each thread block loads a SINGLE Q tile, iterates over all respective KV, so it only cares
    // abt size of Q tile, so use tr for num thread blocks
    // how much sram to allocate?
    size_t kv_size = Bc * d;
    size_t qo_size = Br * d;
    size_t p_size = Br * Bc;
    // shared for K, V, Q, O, m, l
    // don't allocate shared for m and l now
    size_t smem_size = sizeof(half) * (2*kv_size + 2*qo_size + p_size);

    float scaling_factor = (1 / sqrt(d));
    // std::cout << "scaling QK^T by " << (scaling_factor) << "\n";

    std::cout << "Launching kernel with smem size " << smem_size << "\n";

    if (smem_size > props.sharedMemPerBlock) {
        printf("Too much shared memory requested per block\n");
    } else {
        // want flat thread grid
       hipLaunchKernelGGL(( flash_attention_kernel), dim3(tr), dim3(32*N_WARPS), smem_size, 0, 
            (half*)Q.const_data_ptr(), (half*)K.const_data_ptr(), (half*)V.const_data_ptr(), 
            (half*)O.mutable_data_ptr(), 1, N, d, tc);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            printf("Error running kernel: %s\n", hipGetErrorString(err));
        }
    }

    
    return O;
}

void benchmark_flash_attn(int N, int d, int n_iters=100) {

    printf("Running %d iters of flash attn on %d x %d matrices\n", n_iters, N, d);
    
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf);
    torch::Tensor Q = torch::rand({N, d}, options);
    torch::Tensor K = torch::rand({N, d}, options);
    torch::Tensor V = torch::rand({N, d}, options);

    // if I don't re instantiate each time, this will be incorrect
    torch::Tensor O = torch::zeros_like(Q);

    size_t kv_size = Bc * d;
    size_t qo_size = Br * d;
    size_t p_size = Br * Bc;
    // shared for K, V, Q, O, m, l
    // don't allocate shared for m and l now
    size_t smem_size = sizeof(half) * (2*kv_size + 2*qo_size + p_size);
    size_t tc = N / Bc + (N%Bc!=0);
    size_t tr = N / Br + (N%Br!=0);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    for (int i=0; i<n_iters; ++i)
       hipLaunchKernelGGL(( flash_attention_kernel), dim3(tr), dim3(Br), smem_size, 0, 
                (half*)Q.const_data_ptr(), (half*)K.const_data_ptr(), (half*)V.const_data_ptr(), 
                (half*)O.mutable_data_ptr(), 1, N, d, tc);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    printf("%f ms\n", ms);

    hipEventDestroy(start);
    hipEventDestroy(stop);


}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attn", torch::wrap_pybind_function(flash_attention), "flash_attn");
    m.def("run_bench", torch::wrap_pybind_function(benchmark_flash_attn), "get benchmark for flash attn");
}