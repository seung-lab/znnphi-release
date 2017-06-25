#pragma once

#include "znn/types.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/conv/fwd2/problem.hpp"
#include "znn/layer/dimension.hpp"
#include "znn/layer/conv/fwd2/work.hpp"

namespace znn { namespace phi {

class fwd_execute
{
public:

    virtual void execute( float const *,
                          float       *,
                          float const *,
                          float const * ) const = 0;

    virtual long_t flops() const = 0;

    double gflops() const
    {
        return static_cast<double>(flops()) / 1000000000;
    }
};


template< class Problem >
class fwd_execute_t: public fwd_execute
{
private:
    using size   = typename Problem::size;
    using wshape = typename Problem::shapes::weight;
    using ishape = typename Problem::shapes::input;
    using oshape = typename Problem::shapes::output;

    static const long_t ifm_sets = (size::ifm + SIMD_WIDTH - 1) / SIMD_WIDTH;
    static const long_t ofm_sets = Problem::size::ofm_sets;

    using CD = conv_traits<wshape::depth,1,1>;
    using CH = conv_traits<wshape::height,1,1>;
    using CW = conv_traits<wshape::width,1,1>;

    using DT = dimension<size::depth ,ishape::depth ,oshape::depth>;
    using HT = dimension<size::height,ishape::height,oshape::height>;
    using WT = dimension<size::width ,ishape::width ,oshape::width>;

    using head_t = fwd_work< true,SIMD_WIDTH,
                             DT,HT,WT,
                             CD,CH,CW>;

    using mid_t = fwd_work< false,SIMD_WIDTH,
                            DT,HT,WT,
                            CD,CH,CW>;

    using tail_t = fwd_work<(size::ifm<SIMD_WIDTH),
        (size::ifm % SIMD_WIDTH),
        DT,HT,WT, CD,CH,CW>;

    head_t head;
    mid_t  mid ;
    tail_t tail;

    static const long_t W_NEXT_INPUT =
                            wshape::depth * wshape::height * wshape::width *
                            SIMD_WIDTH * SIMD_WIDTH;

    static const long_t W_NEXT_OUTPUT =
                            W_NEXT_INPUT * ifm_sets;

    long_t loop_ifm( float const * i,
                     float       * o,
                     float const * k,
                     float const * b) const
    {
        long_t flops = 0;

        if ( size::ifm >= SIMD_WIDTH )
        {
            flops += head.flops();
            head.execute(i,o,k,b);
            i += Problem::shapes::input::fm_set;
            k += wshape::depth * wshape::height * wshape::width * SIMD_WIDTH * SIMD_WIDTH;
        }

        for ( long_t x = 1; x < (size::ifm/SIMD_WIDTH); ++x )
        {
            flops += mid.flops();
            mid.execute(i,o,k,b);
            i += Problem::shapes::input::fm_set;
            k += wshape::depth * wshape::height * wshape::width * SIMD_WIDTH * SIMD_WIDTH;
        }

        if ( size::ifm % SIMD_WIDTH )
        {
            flops += tail.flops();
            tail.execute(i,o,k,b);
        }

        return flops;
    }

    long_t loop_ofm( float const * i,
                     float       * o,
                     float const * k,
                     float const * b) const
    {
        long_t flops = 0;
        for ( long_t x = 0; x < size::ofm_sets; ++x )
        {
            flops += loop_ifm(i, o, k, b);
            o += Problem::shapes::output::fm_set;
            k += wshape::output;
            b += SIMD_WIDTH;
        }
        return flops;
    }


    void do_exec( float const * i,
                  float       * o,
                  float const * k,
                  float const * b,
                  long_t ifms,
                  long_t ofms,
                  bool   first ) const
    {

        //std::cout << ifms << ' ' << ofms << ' ' << first << '\n';
        if ( (ifms>ofms) && (ifms%2==0) )
        {
            do_exec( i, o, k, b, ifms/2, ofms, first );

            do_exec( i + (ifms/2) * Problem::shapes::input::fm_set,
                     o,
                     k + (ifms/2) * W_NEXT_INPUT,
                     b,
                     ifms/2, ofms, false );
        }
        else if ( (ifms>ofms) && (ifms%3==0) )
        {
            do_exec( i, o, k, b, ifms/3, ofms, first );

            do_exec( i + (ifms/3) * Problem::shapes::input::fm_set,
                     o,
                     k + (ifms/3) * W_NEXT_INPUT,
                     b,
                     ifms/3, ofms, false );

            do_exec( i + (ifms/3) * 2 * Problem::shapes::input::fm_set,
                     o,
                     k + (ifms/3) * 2 * W_NEXT_INPUT,
                     b,
                     ifms/3, ofms, false );
        }
        else if ( ofms%2==0 )
        {
            do_exec( i, o, k, b, ifms, ofms/2, first );

            do_exec( i,
                     o + (ofms/2) * Problem::shapes::output::fm_set,
                     k + (ofms/2) * W_NEXT_OUTPUT,
                     b + (ofms/2) * SIMD_WIDTH,
                     ifms, ofms/2, first );
        }
        else if ( ofms%3==0 )
        {
            do_exec( i, o, k, b, ifms, ofms/3, first );

            do_exec( i,
                     o + (ofms/3) * Problem::shapes::output::fm_set,
                     k + (ofms/3) * W_NEXT_OUTPUT,
                     b + (ofms/3) * SIMD_WIDTH,
                     ifms, ofms/3, first );

            do_exec( i,
                     o + (ofms/3) * 2 * Problem::shapes::output::fm_set,
                     k + (ofms/3) * 2 * W_NEXT_OUTPUT,
                     b + (ofms/3) * 2 * SIMD_WIDTH,
                     ifms, ofms/3, first );
        }
        else if ( ifms%2==0 )
        {
            do_exec( i, o, k, b, ifms/2, ofms, first );

            do_exec( i + (ifms/2) * Problem::shapes::input::fm_set,
                     o,
                     k + (ifms/2) * W_NEXT_INPUT,
                     b,
                     ifms/2, ofms, false );
        }
        else if ( ifms%3==0 )
        {
            do_exec( i, o, k, b, ifms/3, ofms, first );

            do_exec( i + (ifms/3) * Problem::shapes::input::fm_set,
                     o,
                     k + (ifms/3) * W_NEXT_INPUT,
                     b,
                     ifms/3, ofms, false );

            do_exec( i + (ifms/3) * 2 * Problem::shapes::input::fm_set,
                     o,
                     k + (ifms/3) * 2 * W_NEXT_INPUT,
                     b,
                     ifms/3, ofms, false );
        }
        else //if (0)
        {
            if ( first )
            {
                for ( long_t xo = 0; xo < ofms; ++xo )
                {
                    head.execute( i,
                                  o + xo * Problem::shapes::output::fm_set,
                                  k + xo * W_NEXT_OUTPUT,
                                  b + xo * SIMD_WIDTH );
                    for ( long_t xi = 1; xi < ifms; ++xi )
                    {
                        mid.execute( i + xi * Problem::shapes::output::fm_set,
                                      o + xo * Problem::shapes::output::fm_set,
                                      k + xo * W_NEXT_OUTPUT + xi * W_NEXT_INPUT,
                                      b + xo * SIMD_WIDTH );
                    }
                }
            }
            else
            {
                for ( long_t xo = 0; xo < ofms; ++xo )
                {
                    for ( long_t xi = 0; xi < ifms; ++xi )
                    {
                        mid.execute( i + xi * Problem::shapes::output::fm_set,
                                     o + xo * Problem::shapes::output::fm_set,
                                     k + xo * W_NEXT_OUTPUT + xi * W_NEXT_INPUT,
                                     b + xo * SIMD_WIDTH );
                    }
                }
            }
        }
    }

    long_t loop_batch( float const * i,
                       float       * o,
                       float const * k,
                       float const * b) const
    {
        long_t flops = 0;
        for ( long_t x = 0; x < size::batch; ++x )
        {
            //flops += loop_ofm(i, o, k, b);
            do_exec( i, o, k, b, ifm_sets, ofm_sets, true );
            i += Problem::shapes::input::batch;
            o += Problem::shapes::output::batch;
        }
        return flops;
    }

private:
    long_t ioffset_;
    long_t ooffset_;
    long_t koffset_;
    long_t boffset_;

public:

    fwd_execute_t( long_t i, long_t o, long_t k, long_t b )
        : ioffset_(i)
        , ooffset_(o)
        , koffset_(k)
        , boffset_(b)
    {}

    void execute( float const * i,
                  float       * o,
                  float const * k,
                  float const * b) const override
    {
        loop_batch(i + ioffset_,
                   o + ooffset_,
                   k + koffset_,
                   b + boffset_ );
        //std::cout << "Executed flops: " << eflops << "\n";
    }

    virtual long_t flops() const override
    {
        return size::batch * size::ifm * size::ofm_sets * SIMD_WIDTH *
            size::depth * size::height * size::width *
            wshape::depth * wshape::height * wshape::width * 2;
    }

};


}} // namespace znn:phi
