#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/task.hpp"
#include "znn/layer/conv/fwd3/blocking.hpp"
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

    template< long_t RBW >
    typename std::enable_if< RBW==0,void >::type
    chunk( float const * __restrict,
           float       * __restrict,
           float const * __restrict,
           float const * __restrict,
           long_t const * ,
           long_t const * ,
           long_t const * ,
           long_t const *  ) const
    {
    }

    template< long_t RBW >
    typename std::enable_if< (RBW>0),void >::type
    chunk( float const * __restrict i,
           float       * __restrict o,
           float const * __restrict k,
           float const * __restrict b,
           long_t const * __restrict dfrom,
           long_t const * __restrict  dto,
           long_t const * __restrict  hfrom,
           long_t const * __restrict  hto) const

    {
        static signuper s;

        SIMD_FLOAT vout[RBW], vwt; // in registers

#pragma unroll(RBW)
        for ( long_t rbw = 0; rbw < RBW; ++rbw )
        {
            vout[rbw] = conditional_load_or_bias<FIRST>
                ( o + (rbw * W::os), b );
        }



        for ( long_t kd = *dfrom; kd < *dto; ++kd )

        for ( long_t kh = *hfrom; kh < *hto; ++kh )

        for ( long_t s  = 0;  s < SW; ++s  )


#pragma unroll(CW::s)
        for ( long_t kw = 0; kw < CW::s; ++kw )
        {
            vwt = SIMD_LOAD( k +
                             ((kh * CW::s + kw
                               + kd * CW::s * CH::s) * SIMD_WIDTH + s)
                             * SIMD_WIDTH );
#pragma unroll(RBW)
            for ( long_t rbw = 0; rbw < RBW; ++rbw )
            {
                vout[rbw] = SIMD_FMADD
                    ( vwt,
                      SIMD_SET1(i[( (kd * CD::dilation ) * D::is +
                                    (kh * CH::dilation ) * H::is +
                                    (kw * CW::dilation + rbw * CW::stride) * W::is)
                                  + s]),
                      vout[rbw]);
            }
        }

#pragma unroll(RBW)
        for ( long_t rbw = 0; rbw < RBW; ++rbw )
        {
            SIMD_STORE( o + (rbw * W::os), vout[rbw] );
        }
    }


    long_t flops() const
    {
        return CW::s * CH::s * CD::s * SW * D::s * H::s * W::s * SIMD_WIDTH * 2;
    }

    void loopw( float const * __restrict i,
                float       * __restrict o,
                float const * __restrict k,
                float const * __restrict b,
                long_t const * dfrom,
                long_t const * dto,
                long_t const * hfrom,
                long_t const * hto) const
    {
        static const long_t FULL = W::s / RB::w;
        static const long_t PART = W::s % RB::w;

        for ( long_t d = 0; d < FULL; ++d )
        {
            fwd_work::template chunk<RB::w>(i,o,k,b,dfrom,dto,hfrom,hto);
            i += RB::w * W::is * CW::stride;
            o += RB::w * W::os;
        }

        if ( PART )
        {
            fwd_work::template chunk<PART>(i,o,k,b,dfrom,dto,hfrom,hto);
        }
    }

    void looph( float const * __restrict i,
                float       * __restrict o,
                float const * __restrict k,
                float const * __restrict b,
                long_t const * dfrom,
                long_t const * dto,
                long_t const * hfrom,
                long_t const * hto) const
    {
        for ( long_t d = 0; d < H::s; ++d )
        {
            loopw(i,o,k,b,dfrom,dto,hfrom+d,hto+d);
            i += H::is * CH::stride;
            o += H::os;
        }
    }


    void loopd( float const * __restrict i,
                float       * __restrict o,
                float const * __restrict k,
                float const * __restrict b,
                long_t const * dfrom,
                long_t const * dto,
                long_t const * hfrom,
                long_t const * hto ) const
    {
        for ( long_t d = 0; d < D::s; ++d )
        {
            looph(i,o,k,b,dfrom+d,dto+d,hfrom,hto);
            i += D::is * CD::stride;
            o += D::os;
        }
    }

public:
    void execute( float const * i,
                  float       * o,
                  float const * k,
                  float const * b,
                  long_t const * dfrom,
                  long_t const * dto,
                  long_t const * hfrom,
                  long_t const * hto ) const
    {
        loopd(i,o,k,b,dfrom,dto,hfrom,hto);
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
                  float const *,
                  long_t const *,
                  long_t const *,
                  long_t const *,
                  long_t const * ) const
    {}
};


}} // namespace znn:phi
