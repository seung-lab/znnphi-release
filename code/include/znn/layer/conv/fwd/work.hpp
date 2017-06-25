#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/task.hpp"
#include "znn/layer/conv/fwd/blocking.hpp"
#include "znn/util/conditional_load.hpp"

namespace znn { namespace phi {

// kernel shape is
// K[D][H][W][IN_FMAP][OUT_FMAP] with IN=OUT=16

// input and output shapes are
// Im[D][H][W][FMAP]

// for all valid d, h, w, kd, kh, kw, if, of
// Out[d][h][w][of] += K[kd][kh][kw][if][of] * In[d+kd][h+kh][w+kw][if];

template< bool FIRST,                      // load or set to zero
          long_t SW,                       // number of input featuremaps avx
          class D , class H , class W ,    // out size and in/out strides
          class CD, class CH, class CW     // convolution params
          >
class fwd_work
{
private:
    using RB = fwd_blocking_t<D::s,H::s,W::s>;

    struct signuper
    {
        signuper()
        {
            std::cout << "SIGNED: " << RB::d << ' ' << RB::h << ' ' << RB::w << '\n';
        }
    };


public:

    template< long_t RBD, long_t RBH, long_t RBW >
    typename std::enable_if< RBD==0 || RBH==0 || RBW==0,void >::type
    chunk( float const * __restrict,
           float       * __restrict,
           float const * __restrict,
           float const * __restrict ) const
    {
    }

    template< long_t RBD, long_t RBH, long_t RBW >
    typename std::enable_if< (RBD>0) && (RBH>0) && (RBW>0),void >::type
    chunk( float const * __restrict i,
           float       * __restrict o,
           float const * __restrict k,
           float const * __restrict b) const

    {
        static signuper s;

        SIMD_FLOAT vout[RBD][RBH][RBW], vwt; // in registers

#pragma unroll(RBD)
        for ( long_t rbd = 0; rbd < RBD; ++rbd )
#pragma unroll(RBH)
        for ( long_t rbh = 0; rbh < RBH; ++rbh )
#pragma unroll(RBW)
        for ( long_t rbw = 0; rbw < RBW; ++rbw )
        {
            vout[rbd][rbh][rbw] = conditional_load_or_bias<FIRST>
                ( o + (rbw * W::os + rbd * D::os + rbh * H::os), b );
        }


        for ( long_t kd = 0; kd < CD::s; ++kd )

        for ( long_t kh = 0; kh < CH::s; ++kh )

        for ( long_t s  = 0;  s < SW; ++s  )

        for ( long_t kw = 0; kw < CW::s; ++kw )
        {
            vwt = SIMD_LOAD( k +
                             ((kh * CW::s + kw
                               + kd * CW::s * CH::s) * SIMD_WIDTH + s)
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
                      SIMD_SET1(i[( (kd * CD::dilation + rbd * CD::stride) * D::is +
                                    (kh * CH::dilation + rbh * CH::stride) * H::is +
                                    (kw * CW::dilation + rbw * CW::stride) * W::is)
                                  + s]),
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
            SIMD_STORE( o + (rbw * W::os + rbd * D::os + rbh * H::os),
                        vout[rbd][rbh][rbw] );
        }
    }


    long_t flops() const
    {
        return CW::s * CH::s * CD::s * SW * D::s * H::s * W::s * SIMD_WIDTH * 2;
    }

    template< long_t DD, long_t HH >
    void loopw( float const * __restrict i,
                float       * __restrict o,
                float const * __restrict k,
                float const * __restrict b) const
    {
        static const long_t FULL = W::s / RB::w;
        static const long_t PART = W::s % RB::w;

        for ( long_t d = 0; d < FULL; ++d )
        {
            fwd_work::template chunk<DD,HH,RB::w>(i,o,k,b);
            i += RB::w * W::is * CW::stride;
            o += RB::w * W::os;
        }

        if ( PART )
        {
            fwd_work::template chunk<DD,HH,PART>(i,o,k,b);
        }
    }

    template< long_t DD >
    void looph( float const * __restrict i,
                float       * __restrict o,
                float const * __restrict k,
                float const * __restrict b) const
    {
        static const long_t FULL = H::s / RB::h;
        static const long_t PART = H::s % RB::h;

        for ( long_t d = 0; d < FULL; ++d )
        {
            fwd_work::template loopw<DD,RB::h>(i,o,k,b);
            i += RB::h * H::is * CH::stride;
            o += RB::h * H::os;
        }

        if ( PART )
        {
            fwd_work::template loopw<DD,PART>(i,o,k,b);
        }
    }


    void loopd( float const * __restrict i,
                float       * __restrict o,
                float const * __restrict k,
                float const * __restrict b) const
    {
        static const long_t FULL = D::s / RB::d;
        static const long_t PART = D::s % RB::d;

        for ( long_t d = 0; d < FULL; ++d )
        {
            fwd_work::template looph<RB::d>(i,o,k,b);
            i += RB::d * D::is * CD::stride;
            o += RB::d * D::os;
        }

        if ( PART )
        {
            fwd_work::template looph<PART>(i,o,k,b);
        }
    }

public:
    void execute( float const * i,
                  float       * o,
                  float const * k,
                  float const * b) const
    {
        loopd(i,o,k,b);
        //std::cout << "Work\n";
    }
};

template< bool FIRST,                      // load or set to zero
          class D , class H , class W ,    // out size and in/out strides
          class CD, class CH, class CW     // convolution params
          >
class fwd_work<FIRST, 0, D, H, W, CD, CH, CW>
{

public:
    long_t flops() const
    {
        return 0;
    }

    void execute( float const *,
                  float       *,
                  float const *,
                  float const * ) const
    {}
};


}} // namespace znn:phi
