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

// #define DEBUG
// only works for powers of 2
#define ROUND_UP_TO(x, multiple) (((x) + ((multiple) - 1)) & ~((multiple) - 1))

// in flash attn 2, set to either 64 or 128, can basically pick what you want as long as 
// you have the sram for it
// also number of threads per block
// wait HUGE heuristic: flash attn 2 typically uses 4 or 8 warps per thread block
// sizes of QKV tiles are {64, 128} x {64, 128}
// Is this referring to Br and Bc? Maybe this is kBlockM, N in the cuda code
// https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h
// because of sram limitations, I can't afford the same block sizes and thus I should probably allocate less warps per block
// Br should always be at least 32, since many parts of the kernel are row-wise over tiles of height Br
// Furthermore, as long as things are threadwise over Br, Br should always be larger than Bc and generally as large as possible
const size_t Br = 32;
const size_t Bc = 48;
// const size_t Br = 8;
// const size_t Bc = 4;
// Br always multiple of 32
const size_t N_WARPS = Br / 32;
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

// everything has to be 16x16
// Remember that all matmuls you need to perform for flashattn have an M of Br, 
// meaning you can map threads/warps to whole rows and j tile across them for your A
// for b, j do the same thing but padded? kind of
// each row of a needs a col of b with same num of tiles, but might be different num of rows and cols
// ahhh: just tile a and b in 16x16 and then split those tiles across however many warps you got

const int MMA_M = 16;
const int MMA_K = 16;
const int MMA_N = 16;

template <bool transpose_b, bool accum>
__device__
void wmma_matmul(const float16_t* a, const float16_t* b, float16_t* c, size_t m, size_t k, size_t n) {
    // wmma fragments
    auto fraga = rocwmma::fragment<matrix_a, MMA_M, MMA_N, MMA_K, float16_t, row_major>();

    auto fragb = rocwmma::fragment<matrix_b, MMA_M, MMA_N, MMA_K, float16_t>();
    auto fragc = rocwmma::fragment<accumulator, MMA_M, MMA_N, MMA_K, float16_t>();
    auto fragacc = rocwmma::fragment<accumulator, MMA_M, MMA_N, MMA_K, float32_t>();

    // Note that we're assuming the matrices are PERFECTLY TILED by 16x16 matrices, so the sizes
    // of your shared memory better match as well!
    size_t m_tiles = m / MMA_M;
    size_t n_tiles = n / MMA_N;
    size_t k_tiles = k / MMA_K;

    size_t n_warps = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    
    // loop each warp over the m*n tiles of the matrix
    // assume we don't know what shapes we'll get
    for (int tile_idx=warpId; tile_idx<m_tiles*n_tiles; tile_idx+=n_warps) {
        rocwmma::fill_fragment(fragacc, 0.0f);
        // compute the ith output tile of the output matrix
        int i = tile_idx / n_tiles;
        int j = tile_idx % n_tiles;
        // if (threadIdx.x%32 == 0 && blockIdx.x == 0) {
        //     printf("Warp %d computing tile %d of matrix, coords (%d, %d), will do %d loops\n", warpId, tile_idx, i, j, k_tiles);
        // }
        // iterate over ith row of a and jth col of b (both should be length k)
        for (int k_idx=0; k_idx<k_tiles; ++k_idx) {
            // B is in row-major format, but treat it LIKE a col-major matrix so it has the result of AB^T without
            // tranpsosing B first
            // This will allow for easy vectorizing loading of K exactly like V
            int b_idx = transpose_b ? (k_idx*MMA_M + j*MMA_N*k) : (n*k_idx*MMA_M + j*MMA_N);
            auto b_layout = transpose_b ? rocwmma::mem_col_major : rocwmma::mem_row_major;
            // if (threadIdx.x%32 == 0 && blockIdx.x == 0) {
            //     printf("Warp %d loading from idx %d of a and idx %d of b\n", warpId, (k*i*MMA_M + k_idx*MMA_N), b_idx);
            // }
            rocwmma::load_matrix_sync(fraga, a + (k*i*MMA_M + k_idx*MMA_N), k);
            rocwmma::load_matrix_sync(fragb, b + b_idx, transpose_b ? k : n, b_layout);
            rocwmma::synchronize_workgroup();

            rocwmma::mma_sync(fragacc, fraga, fragb, fragacc);
        }

        if constexpr (accum) {
            rocwmma::load_matrix_sync(fragc, c + (n*i*MMA_M + j*MMA_N), n, rocwmma::mem_row_major);
            rocwmma::synchronize_workgroup();
            for (int i=0; i<fragc.num_elements; ++i) 
                fragc.x[i] += fragacc.x[i];
        } else {
            for (int i=0; i<fragc.num_elements; ++i) 
                fragc.x[i] = fragacc.x[i];
        }

        // store result
        rocwmma::store_matrix_sync(c + (n*i*MMA_M + j*MMA_N), fragc, n, rocwmma::mem_row_major);

    }
    rocwmma::synchronize_workgroup();
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
    size_t Bc_padded = ROUND_UP_TO(Bc, 16);
    size_t d_padded = ROUND_UP_TO(d, 16);

    // total size of shared k and v is num of vecs in tile * size of vec
    // pad for 16x16 tiles
    size_t kv_size = Bc_padded * d_padded;
    half *K_s = sram;
    half *V_s = sram + kv_size;

    size_t qo_size = Br * d_padded;
    half *Q_s = V_s + kv_size;
    half *O_s = Q_s + qo_size;

    // {Brxd} @ {dxBc} = {BrxBc}
    size_t p_size = Br * Bc_padded;
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
    size_t rows_from = min(Br, N-qo_idx);
    // assume d always divisible by 2
    // also need to pad with 0's; we know that Br is divisible by 32, so just need to ensure
    // d is divisible by 16
    for (int k=0; k<d_padded/2; ++k) {
        // an index represents TWO floats, since each thread loads in two at a time; 
        // so double idx when you use it
        // these are indices FROM, not to; not for double floats 
        int idx = (blockDim.x * k + threadIdx.x)*2;
        int i = idx / d_padded;
        int j = idx % d_padded; // d_padded is stride for Q and O (padded)

        // padding value for tiled matmul
        if (j >= d || i >= rows_from) {
            reinterpret_cast<half2*>(Q_s)[idx/2] = H2_ZERO;
        // there will only be this bad warp divergence in a single warp
        } else if (j+1 < d) {
            // if you're in one of the padding cols, just load zeros
            reinterpret_cast<half2*>(Q_s)[idx/2] = reinterpret_cast<const half2*>(&Q[qo_idx*d])[i*d/2 + j/2];
        } else {
            // when using idx in the half2 vector, the pointer arithmetic is DOUBLED, so double here for parity
            Q_s[i*d_padded + j] = Q[qo_idx*d + idx/2];
            Q_s[i*d_padded + j+1] = H_ZERO;
        } 
        reinterpret_cast<half2*>(O_s)[idx/2] = H2_ZERO;
    }
    // save the syncthreads for later

    // __syncthreads();
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("Q block %d:\n", blockIdx.x);
    //     tensorprint(Q_s, Br, d_padded);
        // printf("O:\n");
        // tensorprint(O_s, Br, d_padded);
    // }

    int r = blockDim.x / Bc;
    // loop over kv tiles
    int kv_idx;
    // load vectorized to these regs, then place the values in shared
    for (size_t i=0; i<tc; ++i) {

        // Index of the KV vector to start from
        kv_idx = Bc * i;

        // load tile of k and v into sram
        size_t size_to = Bc_padded*d_padded;
        rows_from = min(Bc, N-kv_idx);
        int n_iters = size_to / (blockDim.x*2) + (size_to%(blockDim.x*2) != 0);
        // we can load K EXACTLY as we do V, because we just pretend it's col-major 
        // for the WMMA
        for (int k=0; k<n_iters; ++k) {
            // indices should line up perfectly between to and from, assume 
            // the pointers are to the starts of only what you want to copy (not the potential beginning of the matrix)
            // remember that the DOUBLED index into the matrix
            int idx = (blockDim.x * k + threadIdx.x)*2;
            int i = idx / d_padded;
            int j = idx % d_padded;
            
            if (j >= d || i >= Bc) {
                reinterpret_cast<half2*>(K_s)[idx/2] = H2_MIN;
                reinterpret_cast<half2*>(V_s)[idx/2] = H2_ZERO;
            } else if (j+1 < d) {
                reinterpret_cast<half2*>(K_s)[idx/2] = (i<rows_from) ? reinterpret_cast<const half2*>(&K[kv_idx*d])[i*d/2 + j/2] : H2_MIN;
                reinterpret_cast<half2*>(V_s)[idx/2] = (i<rows_from) ? reinterpret_cast<const half2*>(&V[kv_idx*d])[i*d/2 + j/2] : H2_ZERO;
            } else {
                K_s[i*d_padded + j] =  K[kv_idx*d + idx/2];
                K_s[i*d_padded + j+1] = H_MIN;
                V_s[i*d_padded + j] = V[kv_idx*d + idx/2];
                V_s[i*d_padded + j+1] = H_ZERO;
            }
        }
        
        __syncthreads();

        // if (threadIdx.x == 0 && blockIdx.x==0) {
        //     printf("K block %d:\n", blockIdx.x);
        //     tensorprint(K_s, Bc_padded, d_padded);
        //     printf("V block %d:\n", blockIdx.x);
        //     tensorprint(V_s, Bc_padded, d_padded);
        // }

        // matmul QK^T
        wmma_matmul<true, false>((float16_t*)Q_s, (float16_t*)K_s, (float16_t*)P_s, Br, d_padded, Bc_padded);
        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //         printf("P_s before shifted exp:\n");
        //         tensorprint(P_s, Br, Bc_padded);
        // }

        // define sumexp l_new to divide out at the end
        // no need for shared memory, because row-wise
        float l_new;
        // calculate new maxes
        // for all these row-wise ops, just use the first Br threads per block to minimize warp divergence

        // calculate max over a single row of P_s
        half mt_h = H_MIN;
        for (int i=0; i<Bc; ++i)
            mt_h = max(mt_h, P_s[threadIdx.x*Bc_padded + i]);
        float mt = __half2float(mt_h);
        // shifted exponentiation of each row and row-wise sumexp
        float lt = 0;
        for (int i=0; i<Bc; ++i) {
            float exp = __expf(__half2float(P_s[threadIdx.x*Bc_padded + i]) - mt);
            P_s[threadIdx.x*Bc + i] = __float2half(exp);
            lt += exp;
        }
        // if (threadIdx.x == 0 && blockIdx.x == 0) {
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
            O_s[threadIdx.x*d_padded + i] = __hmul(O_s[threadIdx.x*d_padded + i], __float2half(old_lse));
        
        // only necessary for now
        // __syncthreads();

        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     printf("Oi after getting renomalized:\n");
        //     tensorprint(O_s, Br, d);
        // }
        // scale P's rows by exp(mt - m_new) as in flashattn
        // half undo_exp_h = __float2half(undo_exp);
        for (int i=0; i<Bc; ++i)
            P_s[threadIdx.x*Bc_padded+i] = __hmul(P_s[threadIdx.x*Bc_padded+i], __float2half(undo_exp));
        
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
        wmma_matmul<false, true>((float16_t*)P_s, (float16_t*)V_s, (float16_t*)O_s, Br, Bc_padded, d_padded);
        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("Oi after the matmul:\n");
        //     tensorprint(O_s, Br, d_padded);
        // }

        // divide new O by l_new
        half dividend = __float2half(__fdividef(1, l_new));
        // if (__hisinf(dividend) || __hisnan(dividend))
        //     printf("dividend is inf or nan, l_new is %f\n", l_new);
        // printf("l_new is %f, dividend is %f\n", l_new, dividend);
        for (int i=0; i<d; ++i)
            O_s[threadIdx.x*d_padded + i] = __hmul(O_s[threadIdx.x*d_padded + i], dividend);
            
        // if (threadIdx.x == 0 && blockIdx.x == 1) {
        //     printf("Oi after divided by l_new; end of block:\n");
        //     tensorprint(O_s, Br, d);
        // }


        // don't need to write back to hbm as in the pytorch version, because a thread 
        // block iterates only over a single row of KV instead of the whole attn matrix
    }

    // write tile of O back to global results, ONLY the actual values
    size_t rows_to = min(Br, N-qo_idx);
    for (int k=0; k<d/2; ++k) {
        // indices should line up perfectly between to and from, assume 
        // the pointers are to the starts of only what you want to copy (not the potential beginning of the matrix)
        int idx = (blockDim.x * k + threadIdx.x)*2;
        int i = idx / d_padded;
        int j = idx % d_padded; 
        // maybe still faster to have all threads loading instead of turning off enough to get an even size?
        // we know that we will never exceed the Br width here
        if (j+1 < d) {
            reinterpret_cast<half2*>(&O[qo_idx*d])[i*d/2 + j/2] = reinterpret_cast<half2*>(O_s)[idx/2];
        } else if (j < d) {
            O[qo_idx*d + idx/2] = O_s[i*d_padded + j];
        }
    }

   
    __syncthreads();

}

torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {

    #ifdef DEBUG
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
    #endif
    
    size_t N = Q.size(0);
    size_t d = Q.size(1);

    // thread block size dictates how many vectors per QKV block, and how many thread blocks required

    // // hold up all these values are knowable before runtime
    // // size and num of KV tiles
    size_t tc = N / Bc + (N%Bc!=0);

    // // size and num of Q, O tiles
    size_t tr = N / Br + (N%Br!=0);

    #ifdef DEBUG
    printf("tc=%d, tr=%d\n", tc, tr);
    #endif

    // instantiate extra tensors
    auto options = torch::TensorOptions().device(Q.device()).dtype(Q.dtype()); 
    torch::Tensor O = torch::empty_like(Q);

    // each thread block loads a SINGLE Q tile, iterates over all respective KV, so it only cares
    // abt size of Q tile, so use tr for num thread blocks
    // be sure to pad these s.t. you can tile to 16x16; 
    // both dims must be padded to mul of 16, but size just needs to be mul of 256

    size_t Bc_padded = ROUND_UP_TO(Bc, 16);
    size_t d_padded = ROUND_UP_TO(d, 16);

    size_t kv_size = Bc_padded * d_padded;
    size_t qo_size = Br * d_padded;
    size_t p_size = Br * Bc_padded;
    // shared for K, V, Q, O, m, l
    // don't allocate shared for m and l now
    size_t smem_size = sizeof(half) * (2*kv_size + 2*qo_size + p_size);

    // std::cout << "Launching kernel with smem size " << smem_size << "\n";

    #ifdef DEBUG
    if (smem_size > props.sharedMemPerBlock) {
        printf("Too much shared memory requested per block\n");
    } else {
    #endif
        // want flat thread grid
       hipLaunchKernelGGL(( flash_attention_kernel), dim3(tr), dim3(32*N_WARPS), smem_size, 0, 
            (half*)Q.const_data_ptr(), (half*)K.const_data_ptr(), (half*)V.const_data_ptr(), 
            (half*)O.mutable_data_ptr(), 1, N, d, tc);

    #ifdef DEBUG
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            printf("Error running kernel: %s\n", hipGetErrorString(err));
        }
    }
    #endif

    
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

__global__ void matmul_kernel(const float16_t* A, const float16_t* B, float16_t* C, 
                            size_t m, size_t k, size_t n, bool b_t) { 
    if (b_t) {
        wmma_matmul<true, false>(A, B, C, m, k, n);
    } else {
        wmma_matmul<false, false>(A, B, C, m, k, n);
    }
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B, bool b_t) {

    int m = A.size(0);
    int k = A.size(1);
    int n = b_t ? B.size(0) : B.size(1);

    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf);
    torch::Tensor C = torch::zeros({m, n}, options);
   hipLaunchKernelGGL(( matmul_kernel), dim3(1), dim3(32), 0, 0, 
        (float16_t*)A.const_data_ptr(), (float16_t*)B.const_data_ptr(), 
        (float16_t*)C.mutable_data_ptr(), m, k, n, b_t);
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attn", torch::wrap_pybind_function(flash_attention), "flash_attn");
    m.def("run_bench", torch::wrap_pybind_function(benchmark_flash_attn), "get benchmark for flash attn");
    m.def("matmul", torch::wrap_pybind_function(matmul), "matmul test on 16x16 float16");
}