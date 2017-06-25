#pragma once

#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/task.hpp"
#include "znn/types.hpp"
#include "znn/util/conditional_load.hpp"

namespace znn
{
namespace phi
{

// D<s, is, os>
// H<s, is, os>

template <class D, class H, class W, long_t KD, long_t KH, long_t KW,
          long_t NEXT>
class mpf_forward_task : virtual public task
{
private:
    float const* __restrict input;
    float* __restrict output;

public:
    std::unique_ptr<task> offset_copy(long_t ioff, long_t ooff,
                                      long_t) const override
    {
        return std::make_unique<mpf_forward_task>(input + ioff, output + ooff);
    }

public:
    mpf_forward_task(float const* __restrict i, float* __restrict o)
        : input(i)
        , output(o)
    {
    }

    inline void one_position(float const* __restrict i,
                             float* __restrict o) const
    {
        SIMD_FLOAT neg_infinity = SIMD_SET1(static_cast<float>(-1e10));

#pragma unroll(KD)
        for (long_t bd = 0; bd < KD; ++bd)
#pragma unroll(KH)
            for (long_t bh = 0; bh < KH; ++bh)
#pragma unroll(KW)
                for (long_t bw = 0; bw < KW; ++bw)
                {
                    SIMD_FLOAT m = neg_infinity;

#pragma unroll(KD)
                    for (long_t kd = 0; kd < KD; ++kd)
#pragma unroll(KH)
                        for (long_t kh = 0; kh < KH; ++kh)
#pragma unroll(KW)
                            for (long_t kw = 0; kw < KW; ++kw)
                            {
                                m = SIMD_MAX(m, SIMD_LOAD(i +
                                                          ((bd + kd) * D::is +
                                                           (bh + kh) * H::is +
                                                           (bw + kw) * W::is) *
                                                              SIMD_WIDTH));
                            }

                    SIMD_STREAM(o + (bd * KH * KW + bh * KW + bw) * NEXT, m);
                }
    }

    long_t flops() const override
    {
        return D::s * H::s * W::s * KD * KD * KH * KH * KW * KW * SIMD_WIDTH;
    }

    void execute() const override
    {
        for (long_t d = 0; d < D::s; ++d)
            for (long_t h = 0; h < H::s; ++h)
                for (long_t w = 0; w < W::s; ++w)
                {
                    one_position(
                        input +
                            (d * KD * D::is + h * KH * H::is + w * KW * W::is) *
                                SIMD_WIDTH,
                        output +
                            (d * D::os + h * H::os + w * W::os) * SIMD_WIDTH);
                }
    }
};
}
} // namespace znn:phi
