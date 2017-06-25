#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/task.hpp"
#include "znn/layer/dimension.hpp"
#include "znn/util/conditional_load.hpp"

namespace znn { namespace phi {

// kernel shape is
// K[IN_FMAP][D][H][W][OUT_FMAP] with OUT=16, IN <= 16

// input and output shapes are
// Im[D][H][W][FMAP]


// K[if][kd][kh][kw][of] += Out[d][h][w][of] * In[d+kd][h+kh][w+kw][if];

template< long_t SW,                       // number of input featuremaps avx
          class D , class H , class W ,    // out size and in/out strides
          class CD, class CH, class CW     // convolution params
          >
class naive_upd_task
    : virtual public task
{
private:
    float const * __restrict input;
    float const * __restrict output;
    float       * __restrict weight;

public:
    std::unique_ptr<task>
    offset_copy( long_t ioff, long_t ooff, long_t koff ) const override
    {
        return std::make_unique<naive_upd_task>( input + ioff,
                                                 output + ooff,
                                                 weight + koff );
    }

public:

    naive_upd_task( float const * __restrict i,
                    float const * __restrict o,
                    float       * __restrict w )
        : input(i)
        , output(o)
        , weight(w)
    {}

    void chunk( float const * __restrict i,
                float const * __restrict o,
                float       * __restrict k ) const
    {
        for ( long_t ifm = 0; ifm < SW        ; ++ifm )
        for ( long_t kd  = 0; kd  < CD::s     ; ++kd  )
        for ( long_t kh  = 0; kh  < CH::s     ; ++kh  )
        for ( long_t kw  = 0; kw  < CW::s     ; ++kw  )
        for ( long_t ofm = 0; ofm < SIMD_WIDTH; ++ofm )
        {
            // K[ifm][kd][kh][kw][ofm] = 0;
            long_t kidx = ifm * CD::s * CH::s * CW::s * SIMD_WIDTH +
                kd * CH::s * CW::s * SIMD_WIDTH +
                kh * CW::s * SIMD_WIDTH +
                kw * SIMD_WIDTH +
                ofm;

            k[kidx] = 0;

            for ( long_t d = 0; d < D::s; ++d )
            for ( long_t h = 0; h < H::s; ++h )
            for ( long_t w = 0; w < W::s; ++w )
            {
                long_t iidx = ((d+kd) * D::is + (h+kh) * H::is + (w+kw) * W::is) * SIMD_WIDTH + ifm;
                long_t oidx = (d * D::os + h * H::os + w * W::os) * SIMD_WIDTH + ofm;
                k[kidx] += i[iidx] * o[oidx];
            }

        }
    }

    long_t flops() const override
    {
        return CW::s * CH::s * CD::s * SW * D::s * H::s * W::s * SIMD_WIDTH * 2;
    }

    void execute() const override
    {
        chunk(input,output,weight);
    }
};



}} // namespace znn:phi
