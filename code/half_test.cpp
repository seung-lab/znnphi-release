#include <cassert>
#include <chrono>
#include <iostream>
#include <type_traits>
#include <x86intrin.h>
//#include <znn/intrin.hpp>


#if defined(ZNN_AVX512)

#define SIMD_WIDTH 16
#define SIMD_HALF_WIDTH 8

#define SIMD_FMADD _mm512_fmadd_ps
#define SIMD_SET1 _mm512_set1_ps
#define SIMD_LOAD _mm512_load_ps
#define SIMD_STORE _mm512_store_ps
#define SIMD_STREAM _mm512_stream_ps
#define SIMD_ZERO _mm512_setzero_ps

#define SIMD_HALF_LOAD _mm

#define SIMD_MAX _mm512_max_ps

#define SIMD_FLOAT __m512
#define SIMD_HALF_FLOAT __m256

#define SIMD_SINGLE_TO_HALF _mm512_cvtps_ph
#define SIMD_HALF_TO_SINGLE _mm512_cvtph_ps

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
   // __m128i shufmask =  _mm_set_epi8(13,12, 15,14,  9,8, 11,10,  5,4, 7,6,  1,0, 3,2);
    __m256i ourmask = _mm256_set_epi8(29,28, 31,30, 25,24, 27,26, 21,20, 23,22, 17,16, 19,18, 13,12, 15,14,  9,8, 11,10,  5,4, 7,6,  1,0, 3,2);

    __m512 vout[RBD][RBH][RBW], vwt; // in registers
                                     // std::cout << "in chunk \n" ;
#pragma unroll(RBD)
    for (int rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
        for (int rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
            for (int rbw = 0; rbw < RBW; ++rbw)
            {
                vout[rbd][rbh][rbw] =
                    _mm512_cvtph_ps(_mm256_castps_si256(
                    _mm256_load_ps(o + (rbw + rbd * 100 + rbh * 10) * SIMD_WIDTH/2)));
            }

    for (int kd = 0; kd < 1; ++kd)
        for (int kh = 0; kh < 3; ++kh)
        {
            for (int s = 0; s < SIMD_WIDTH/2; ++s)
                for (int kw = 0; kw < 3; ++kw)
                {
                    vwt = _mm512_cvtph_ps(_mm256_castps_si256(
                      _mm256_load_ps(
                        k +
                        ((kh * 3 + kw + kd * 3 * 3) * SIMD_WIDTH + s*2) *
                            SIMD_WIDTH/2)));

#pragma unroll(RBD)
                    for (int rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                        for (int rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                            for (int rbw = 0; rbw < RBW; ++rbw)
                            {
                                 auto second_arg = _mm512_cvtph_ps(
                                    _mm256_shuffle_epi8(
                                    _mm256_castps_si256(
                                    _mm256_set1_ps(
                                        i[((kd + rbd) * 144 + (kh + rbh) * 12 +
                                           (kw + rbw)) * SIMD_WIDTH +
                                          2*s])),ourmask));

                                vout[rbd][rbh][rbw] = SIMD_FMADD(
                                    vwt,
                                    second_arg,
                                    vout[rbd][rbh][rbw]);
                            }
                }

   for (int s = 0; s < SIMD_WIDTH/2; ++s)
                for (int kw = 0; kw < 3; ++kw)
                {
                    vwt = _mm512_cvtph_ps(_mm256_castps_si256(
                      _mm256_load_ps(
                        k +
                        ((kh * 3 + kw + kd * 3 * 3) * SIMD_WIDTH + s*2 + 1) *
                            SIMD_WIDTH/2)));

#pragma unroll(RBD)
                    for (int rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                        for (int rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                            for (int rbw = 0; rbw < RBW; ++rbw)
                            {
                                 auto second_arg = _mm512_cvtph_ps(
                                    _mm256_shuffle_epi8(
                                    _mm256_castps_si256(
                                    _mm256_set1_ps(
                                        i[((kd + rbd) * 144 + (kh + rbh) * 12 +
                                           (kw + rbw)) * SIMD_WIDTH +
                                          2*s+1])),ourmask));

                                vout[rbd][rbh][rbw] = SIMD_FMADD(
                                    vwt,
                                    second_arg,
                                    vout[rbd][rbh][rbw]);
                            }
                }
        }

#pragma unroll(RBD)
    for (int rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
        for (int rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
            for (int rbw = 0; rbw < RBW; ++rbw)
            {
                _mm256_store_ps(o + (rbw + rbd * 100 + rbh * 10) * SIMD_WIDTH/2,
                                _mm256_castsi256_ps(_mm512_cvtps_ph(vout[rbd][rbh][rbw],0)));
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

    float* a =
        (float*)Amalloc(sizeof(float) * 64 * 64 * 64); // new float[64*64*64];
    float* b =
        (float*)Amalloc(sizeof(float) * 64 * 64 * 64); // new float[64*64*64];
    float* c =
        (float*)Amalloc(sizeof(float) * 64 * 64 * 64); //  new float[64*64*64];

    //  	float * a;
    //	float * b;
    //  	float * c;

    #define RBZ 10

    auto begin = std::chrono::high_resolution_clock::now();
    int  iter  = 2000000;
    for (int i = 0; i < iter; i++)
        chunk<1, 1, RBZ>(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    double secs = static_cast<double>(duration) / 1000000;
    double gflops =
        static_cast<double>(2* SIMD_WIDTH * 2 * RBZ * 3 * 3 * SIMD_WIDTH * 2) /
        1000000000;

    std::cout << "gflops is " << (gflops * iter / secs) << std::endl;
    std::cout << b[4] << std::endl;

    Afree(a);
    Afree(b);
    Afree(c);
}
