#pragma once

#include "znn/layer/conv/fwd2/schedule.hpp"
#include "znn/util/kernel_launcher.hpp"

namespace znn { namespace phi {

template< long_t Threads,
          long_t BS,
          long_t OFM, long_t IFM,
          long_t IFD, long_t IFH, long_t IFW,
          long_t OFD, long_t OFH, long_t OFW,
          long_t CD , long_t CH , long_t CW >
class fwd_plan
{
private:
    static const long_t OFM_SETS = (OFM+SIMD_WIDTH-1)/SIMD_WIDTH;
    static const long_t IFM_SETS = (IFM+SIMD_WIDTH-1)/SIMD_WIDTH;

    using ishape_t = fwd_ioshape_t< IFM_SETS * SIMD_WIDTH * IFD * IFH * IFW,
                                    SIMD_WIDTH * IFD * IFH * IFW,
                                    SIMD_WIDTH * IFH * IFW,
                                    SIMD_WIDTH * IFW,
                                    SIMD_WIDTH >;

    using oshape_t = fwd_ioshape_t< OFM_SETS * SIMD_WIDTH * OFD * OFH * OFW,
                                    SIMD_WIDTH * OFD * OFH * OFW,
                                    SIMD_WIDTH * OFH * OFW,
                                    SIMD_WIDTH * OFW,
                                    SIMD_WIDTH >;

    using wshape_t = fwd_wshape_t< IFM_SETS * CD * CH * CW * SIMD_WIDTH * SIMD_WIDTH,
                                   CD, CH, CW >;

    using shape_t = fwd_shapes_t< ishape_t, oshape_t, wshape_t >;

    using problem_t = fwd_problem_t< Threads,
                                     fwd_problem_size_t< BS, IFM, OFM_SETS, OFD, OFH, OFW >,
                                     0, 0, 0, 0, shape_t >;

    exec_vector ev;
    kernel_launcher * launcher;
    std::vector<std::function<void()>> fns;

    void execute_ith( long_t x,
                      float const * i,
                      float       * o,
                      float const * k,
                      float const * b)
    {
        for ( auto & f: ev[x] )
        {
            f->execute(i,o,k,b);
        }
    }

public:
    fwd_plan( kernel_launcher * l,
              float const * i,
              float       * o,
              float const * k,
              float const * b)
        : ev(Threads)
        , launcher(l)
        , fns(Threads)
    {
        fwd_schedule_t<problem_t>::schedule(0,ev);
        for ( long_t x = 0; x < Threads; ++x )
        {
            fns[x] = std::bind( &fwd_plan::execute_ith, this,
                                x, i, o, k, b );
        }
    }

    void execute()
    {
        launcher->launch( &(fns[0]) );
        //launcher->report();
    }

    double gflops() const
    {
        double ret = 0;
        for ( long_t i = 0; i < Threads; ++i )
        {
            for ( auto & x: ev[i] )
            {
                ret += x->gflops();
            }
        }

        return ret;
    }

};



}} // namespace znn:phi
