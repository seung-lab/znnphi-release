#include <cassert>
#include <chrono>
#include <iostream>
#include <type_traits>
#include <x86intrin.h>
#include <functional>
#include <vector>
#include <znn/util/kernel_launcher.hpp>
//#include <znn/intrin.hpp>


#if defined(ZNN_AVX512)

#define SIMD_WIDTH 16

#define SIMD_FMADD _mm512_fmadd_ps
#define SIMD_SET1 _mm512_set1_ps
#define SIMD_LOAD _mm512_load_ps
#define SIMD_STORE _mm512_store_ps
#define SIMD_STREAM _mm512_stream_ps
#define SIMD_ZERO _mm512_setzero_ps

#define SIMD_MAX _mm512_max_ps

#define SIMD_FLOAT __m512

#define SIMD_MAX_BLOCK 31
#define SIMD_W_BLOCK 12

#define SIMD_NUM_REGISTERS 32

#define SIMD_CMP(a, b) _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ)
#define SIMD_MASK_ADD(r, m, a, b) _mm512_mask_add_ps(r, m, a, b)

#elif defined(ZNN_KNC)

#include <type_traits>

namespace std
{
template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
}

#define SIMD_WIDTH 16

#define SIMD_FMADD _mm512_fmadd_ps
#define SIMD_SET1 _mm512_set1_ps
#define SIMD_LOAD _mm512_load_ps
#define SIMD_STORE _mm512_store_ps
#define SIMD_STREAM _mm512_stream_ps
#define SIMD_ZERO _mm512_setzero_ps

#define SIMD_MAX _mm512_max_ps

#define SIMD_FLOAT __m512

#define SIMD_MAX_BLOCK 14
#define SIMD_W_BLOCK 12

#define SIMD_NUM_REGISTERS 32

#define SIMD_CMP(a, b) _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ)
#define SIMD_MASK_ADD(r, m, a, b) _mm512_mask_add_ps(r, m, a, b)

#elif defined(ZNN_AVX2)

#define SIMD_WIDTH 8

#define SIMD_MUL = _mm256_mul_ps
#define SIMD_FMADD _mm256_fmadd_ps
#define SIMD_FNMADD _mm256_fnmadd_ps
#define SIMD_FMSUB _mm256_fmsub_ps
#define SIMD_FNMSUB _mm256_fnmsub_ps
#define SIMD_SET1 _mm256_set1_ps
#define SIMD_LOAD _mm256_load_ps
#define SIMD_STORE _mm256_store_ps
#define SIMD_STREAM _mm256_stream_ps
#define SIMD_ZERO _mm256_setzero_ps

#define SIMD_MAX _mm256_max_ps

#define SIMD_FLOAT __m256

// TODO FIGURE OUT WHY ICC LIKES 7 AND GCC large numbers

#define SIMD_MAX_BLOCK 7
#define SIMD_W_BLOCK 7

#define SIMD_NUM_REGISTERS 16

#elif defined(ZNN_AVX)

#define SIMD_WIDTH 8

#define SIMD_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define SIMD_SET1 _mm256_set1_ps
#define SIMD_LOAD _mm256_load_ps
#define SIMD_STORE _mm256_store_ps
#define SIMD_STREAM _mm256_stream_ps
#define SIMD_ZERO _mm256_setzero_ps

#define SIMD_MAX _mm256_max_ps

#define SIMD_FLOAT __m256

#define SIMD_MAX_BLOCK 7
#define SIMD_W_BLOCK 7

#define SIMD_NUM_REGISTERS 16

#elif defined(ZNN_SSE)

#define SIMD_WIDTH 4

#define SIMD_FMADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define SIMD_SET1 _mm_set1_ps
#define SIMD_LOAD _mm_load_ps
#define SIMD_STORE _mm_store_ps
#define SIMD_STREAM _mm_stream_ps
#define SIMD_ZERO _mm_setzero_ps

#define SIMD_MAX _mm_max_ps

#define SIMD_FLOAT __m128

#define SIMD_MAX_BLOCK 7
#define SIMD_W_BLOCK 7

#define SIMD_NUM_REGISTERS 16

#else

#error "NEED SOME AVX DEFINED"

#endif


template <int RBD, int RBH, int RBW>
void chunk(float const* __restrict i, float* __restrict o,
           float const* __restrict k)
{

    SIMD_FLOAT vout[RBD][RBH][RBW], vwt; // in registers
                                         // std::cout << "in chunk \n" ;
#pragma unroll(RBD)
    for (int rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
        for (int rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
            for (int rbw = 0; rbw < RBW; ++rbw)
            {
                vout[rbd][rbh][rbw] =
                    SIMD_LOAD(o + (rbw  + rbd * 100 + rbh * 10) * SIMD_WIDTH);
            }

    for (int kd = 0; kd < 1; ++kd)
        for (int kh = 0; kh < 3; ++kh)
            for (int s = 0; s < SIMD_WIDTH; ++s)
                for (int kw = 0; kw < 3; ++kw)
                {
                    vwt = SIMD_LOAD(
                        k +
                        ((kh * 3 + kw + kd * 3 * 3) * SIMD_WIDTH + s) *
                            SIMD_WIDTH);

#pragma unroll(RBD)
                    for (int rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                        for (int rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                            for (int rbw = 0; rbw < RBW; ++rbw)
                            {
                                vout[rbd][rbh][rbw] = SIMD_FMADD(
                                    vwt,
                                    SIMD_SET1(
                                        i[((kd + rbd) * 12*12 * SIMD_WIDTH + (kh + rbh) * 12 * SIMD_WIDTH +
                                           (kw + rbw) * SIMD_WIDTH) +
                                          s]),
                                    vout[rbd][rbh][rbw]);
                            }
                }

#pragma unroll(RBD)
    for (int rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
        for (int rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
            for (int rbw = 0; rbw < RBW; ++rbw)
            {
                SIMD_STORE(o + (rbw + rbd * 100 + rbh * 10) * SIMD_WIDTH,
                           vout[rbd][rbh][rbw]);
            }
}

inline void Afree(void* p)
{
#if 1
    std::free(((void**)p)[-1]);
#else
    std::free(p);
#endif
}

inline void* Amalloc(size_t required_bytes)
{
#if 1
    if (required_bytes == 0)
    {
        return nullptr;
    }

    void*  p1; // original block
    void** p2; // aligned block

    size_t alignment = 64;

    int offset = alignment - 1 + sizeof(void*);

    if ((p1 = (void*)std::malloc(required_bytes + offset)) == NULL)
    {
        // DIE("std::bad_alloc()");
        assert(false);
    }

    p2     = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
#else
    void* ret = std::malloc(required_bytes);
    if (ret == NULL)
    {
        DIE("std::bad_alloc()");
        // throw std::bad_alloc();
    }
#endif
}

int main()
{
    std::vector<std::function<void()>> tasks(64*1);

    znn::phi::kernel_launcher launcher(64, 1);

    for ( int i = 0; i < 64; ++i )
    {
        tasks[i] = []() {


    float* a =
        (float*)Amalloc(sizeof(float) * 1264 * 64 * 64); // new float[64*64*64];
    float* b =
        (float*)Amalloc(sizeof(float) * 1264 * 64 * 64); // new float[64*64*64];
    float* c =
        (float*)Amalloc(sizeof(float) * 1264 * 64 * 64); //  new float[64*64*64];

    //  	float * a;
    //	float * b;
    //  	float * c;

    #define RBZ 12

    auto begin = std::chrono::high_resolution_clock::now();
    int  iter  = 2000000;
    for (int i = 0; i < iter; i++)
        chunk<1, 2, RBZ>(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    double secs = static_cast<double>(duration) / 1000000;
    double gflops =
        static_cast<double>(SIMD_WIDTH * 2 * 2 * RBZ * 3 * 3 * SIMD_WIDTH) /
        1000000000;

    std::cout << "gflops is " << (gflops * iter / secs) << std::endl;
    std::cout << b[4] << std::endl;

    //Afree(a);
    //Afree(b);
    //Afree(c);
        };

    }

    launcher.launch(tasks.data());
    // Afree(a);
    // Afree(b);
    // Afree(c);
}
