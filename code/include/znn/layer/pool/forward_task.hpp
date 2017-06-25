#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/task.hpp"
#include "znn/util/conditional_load.hpp"

namespace znn { namespace phi {



// kernel shape is
// K[D][H][W][IN_FMAP][OUT_FMAP] with IN=OUT=16

// input and output shapes are
// Im[D][H][W][FMAP]

// for all valid d, h, w, kd, kh, kw, if, of
// Out[d][h][w][of] += K[kd][kh][kw][if][of] * In[d+kd][h+kh][w+kw][if];

template< bool ZERO, class D, class H, class W,
          long_t KD, long_t KH, long_t KW, long_t SW,
          class RB
          >
class forward_task
    : virtual public task
{
private:
    float const * __restrict input;
    float       * __restrict output;
    float const * __restrict weight;

public:
    std::unique_ptr<task>
    offset_copy( long_t ioff, long_t ooff, long_t koff ) const override
    {
        return std::make_unique<forward_task>( input + ioff,
                                               output + ooff,
                                               weight + koff );
    }

public:

    forward_task( float const * __restrict i,
                  float       * __restrict o,
                  float const * __restrict w )
        : input(i)
        , output(o)
        , weight(w)
    {}

    template< long_t RBD, long_t RBH, long_t RBW >
    void chunk( float const * __restrict i,
                float       * __restrict o ) const
    {
        SIMD_FLOAT vout[RBD][RBH][RBW], vwt; // in registers

#pragma unroll(RBD)
        for ( long_t rbd = 0; rbd < RBD; ++rbd )
#pragma unroll(RBH)
        for ( long_t rbh = 0; rbh < RBH; ++rbh )
#pragma unroll(RBW)
        for ( long_t rbw = 0; rbw < RBW; ++rbw )
        {
            vout[rbd][rbh][rbw] = conditional_load<ZERO>
                ( o + (rbw * W::os + rbd * D::os + rbh * H::os) * SIMD_WIDTH );
        }

        for ( long_t kd = 0; kd < KD; ++kd )

        for ( long_t kh = 0; kh < KH; ++kh )

        for ( long_t kw = 0; kw < KW; ++kw )

        for ( long_t s  = 0;  s < SW; ++s  )
        {
            vwt = SIMD_LOAD( weight +
                             ((kh * KW + kw
                               + kd * KW * KH) * SIMD_WIDTH + s)
                             * SIMD_WIDTH );

#pragma unroll(RBD)
            for ( long_t rbd = 0; rbd < RBD; ++rbd )
#pragma unroll(RBH)
            for ( long_t rbh = 0; rbh < RBH; ++rbh )
#pragma unroll(RBW)
            for ( long_t rbw = 0; rbw < RBW; ++rbw )
            {
                vout[rbd][rbh][rbw] = SIMD_FMADD
                    ( vwt,
                      SIMD_SET1(i[( (kd+rbd) * D::is +
                                    (kh+rbh) * H::is +
                                    (kw+rbw) * W::is) *
                                  SIMD_WIDTH + s]),
                      vout[rbd][rbh][rbw]);
            }
        }


#pragma unroll(RBD)
        for ( long_t rbd = 0; rbd < RBD; ++rbd )
#pragma unroll(RBH)
        for ( long_t rbh = 0; rbh < RBH; ++rbh )
#pragma unroll(RBW)
        for ( long_t rbw = 0; rbw < RBW; ++rbw )
        {
            SIMD_STORE( o + (rbw * W::os + rbd * D::os + rbh * H::os)
                        * SIMD_WIDTH,
                        vout[rbd][rbh][rbw] );
        }
    }

    long_t flops() const override
    {
        return KW * KH * KD * SW * W::s * SIMD_WIDTH * 2;
    }

    template< long_t DD, long_t HH >
    void loopw( float const * __restrict i,
                float       * __restrict o ) const
    {
        static const long_t FULL = W::s / RB::w;
        static const long_t PART = W::s % RB::w;

        for ( long_t d = 0; d < FULL; ++d )
        {
            forward_task::template chunk<DD,HH,RB::w>(i,o);
            i += RB::w * W::is * SIMD_WIDTH;
            o += RB::w * W::os * SIMD_WIDTH;
        }

        if ( PART )
        {
            forward_task::template chunk<DD,HH,PART>( i, o );
        }
    }

    template< long_t DD >
    void looph( float const * __restrict i,
                float       * __restrict o ) const
    {
        static const long_t FULL = H::s / RB::h;
        static const long_t PART = H::s % RB::h;

        for ( long_t d = 0; d < FULL; ++d )
        {
            forward_task::template loopw<DD,RB::h>(i,o);
            i += RB::h * H::is * SIMD_WIDTH;
            o += RB::h * H::os * SIMD_WIDTH;
        }

        if ( PART )
        {
            forward_task::template loopw<DD,PART>( i, o );
        }
    }


    void loopd( float const * __restrict i,
                float       * __restrict o ) const
    {
        static const long_t FULL = D::s / RB::d;
        static const long_t PART = D::s % RB::d;

        for ( long_t d = 0; d < FULL; ++d )
        {
            forward_task::template looph<RB::d>(i,o);
            i += RB::d * D::is * SIMD_WIDTH;
            o += RB::d * D::os * SIMD_WIDTH;
        }

        if ( PART )
        {
            forward_task::template looph<PART>( i, o );
        }
    }

    void execute() const override
    {
        loopd(input,output);
    }
};



}} // namespace znn:phi
