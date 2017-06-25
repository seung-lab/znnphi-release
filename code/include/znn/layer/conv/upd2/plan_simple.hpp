#pragma once

//#include "znn/layer/conv/fwd/schedule.hpp"
//#include "znn/util/kernel_launcher.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/layer/conv/upd/util.hpp"
#include "znn/layer/conv/upd/problem.hpp"
#include "znn/layer/conv/upd/split.hpp"
#include "znn/layer/conv/upd/work.hpp"
#include "znn/layer/dimension.hpp"
#include "znn/util/kernel_launcher.hpp"
#include <vector>

#define PRINT_CONST( c )                        \
    std::cout << #c << ": " << c << std::endl

namespace znn { namespace phi {

template< long_t Threads,
          long_t BS,
          long_t OFM, long_t IFM,
          long_t IFD, long_t IFH, long_t IFW,
          long_t OFD, long_t OFH, long_t OFW,
          long_t CD , long_t CH , long_t CW >
class upd_plan
{
private:
    static const long_t OFM_SETS = (OFM+SIMD_WIDTH-1)/SIMD_WIDTH;
    static const long_t IFM_SETS = (IFM+SIMD_WIDTH-1)/SIMD_WIDTH;

    static const long_t OFM_STRIDE = OFD * OFH * OFW * SIMD_WIDTH;
    static const long_t IFM_STRIDE = IFD * IFH * IFW * SIMD_WIDTH;

    static const long_t W_IN_STRIDE  = CD * CH * CW * SIMD_WIDTH * SIMD_WIDTH;
    static const long_t W_OUT_STRIDE = W_IN_STRIDE * IFM_SETS;

    static const long_t I_STRIDE = IFM_STRIDE * IFM_SETS;
    static const long_t O_STRIDE = OFM_STRIDE * OFM_SETS;

    static const long_t K_STRIDE = W_OUT_STRIDE * OFM_SETS; // for workspace


    // Kernel elements per pair
    static const long_t KERNEL_PAIR_ELEMENTS =
        CD * CH * CW * SIMD_WIDTH * SIMD_WIDTH;

    // All kernel strides
    static const long_t KERNEL_NEXT_INPUT  = KERNEL_PAIR_ELEMENTS;
    static const long_t KERNEL_NEXT_OUTPUT = KERNEL_NEXT_INPUT * IFM_SETS;

    static const long_t INPUT_NEXT_BATCH  = IFM_SETS * SIMD_WIDTH * IFD * IFH * IFD;
    static const long_t OUTPUT_NEXT_BATCH = OFM_SETS * SIMD_WIDTH * OFD * OFH * OFD;

    void collect_kernel( float const * __restrict w,
                         float       * __restrict k,
                         float rate,
                         long_t from,
                         long_t to )
    {
        SIMD_FLOAT eta = SIMD_SET1(rate);

        for (long_t y = from; y < to; ++y )
        {
            SIMD_FLOAT a = SIMD_LOAD(k + y * SIMD_WIDTH);
            SIMD_FLOAT b = SIMD_LOAD(w + y * SIMD_WIDTH);
            a = SIMD_FMADD(b,eta,a);
            SIMD_STORE(k + y * SIMD_WIDTH, a);
        }
    }

    void schedule_collect( float const * __restrict w,
                           float       * __restrict k,
                           float rate )
    {
        static const long_t total_kernel_elements
            = CD * CH * CW * SIMD_WIDTH * IFM_SETS * OFM_SETS;

        static const long_t k_per_thread
            = (total_kernel_elements+Threads-1) / Threads;

        for ( long_t t = 0; t < Threads; ++t )
        {
            reduce_fns[t] = [=]() {
                collect_kernel( w, k, rate,
                                t * k_per_thread,
                                std::min( (t+1) * k_per_thread, total_kernel_elements) );
            };
            std::cout << "Thread[ " << t << " ] FROM: " << t * k_per_thread
                      << " TO " << std::min( (t+1) * k_per_thread, total_kernel_elements)
                      << "\n";
        }
    }

private:
    std::vector<upd_problem_args> pair_split( long_t ifm, long_t ofm,
                                              long_t ioff, long_t ooff,
                                              long_t koff ) const
    {
        if ( ( ofm % 2 ) == 0 )
        {
            std::vector<upd_problem_args> ret
                = pair_split( ifm, ofm/2, ioff, ooff, koff );

            std::vector<upd_problem_args> rest
                = pair_split( ifm, ofm/2, ioff,
                              ooff + (ofm/2) * OFM_STRIDE,
                              koff + (ofm/2) * W_OUT_STRIDE );

            for ( const auto & e: rest ) ret.push_back(e);

            return ret;
        }
        else if ( ( ifm % 2 ) == 0 )
        {
            std::vector<upd_problem_args> ret
                = pair_split( ifm/2, ofm, ioff, ooff, koff );

            std::vector<upd_problem_args> rest
                = pair_split( ifm/2, ofm,
                              ioff + (ifm/2) * IFM_STRIDE,
                              ooff,
                              koff + (ifm/2) * W_IN_STRIDE );

            for ( const auto & e: rest ) ret.push_back(e);

            return ret;
        }
        else
        {
            std::vector<upd_problem_args> ret;

            for ( long_t o = 0; o < ofm; ++o )
            {
                for ( long_t i = 0; i < ifm; ++i )
                {
                    ret.push_back(upd_problem_args( ioff + i * IFM_STRIDE,
                                                    ooff + o * OFM_STRIDE,
                                                    koff + i * W_IN_STRIDE
                                                    + i * W_OUT_STRIDE ));
                }
            }

            return ret;
        }
    }

    kernel_launcher * launcher;

    std::vector<std::function<void()>> conv_fns;
    std::vector<std::function<void()>> reduce_fns;


    using work_type = upd_work<
        dimension<OFD,IFH*IFW,OFH*OFW>,
        dimension<OFH,IFW,OFW>,
        dimension<OFW,1,1>,
        conv_traits<CD,1,1>,
        conv_traits<CH,1,1>,
        conv_traits<CW,1,1>>;

private:


    void schedule_threads( float const * i,
                           float const * o,
                           float *       w )
    {
        long_t per_thread = (this->pairs.size() + Threads - 1) / Threads;


        for ( long_t tno = 0; tno < Threads; ++tno )
        {
            std::cout << "\tScheduling thread " << tno
                      << " from: " << ( tno * per_thread )
                      << "  to " << std::min( tno * per_thread + per_thread,
                                              static_cast<long_t>(this->pairs.size()) )
                      << "\n";

            conv_fns[tno] = [this,i,o,w,per_thread,tno]() {
                for ( long_t b = 0; b < BS; ++b )
                {
                    for ( long_t io = tno * per_thread;
                          io < std::min( tno * per_thread + per_thread,
                                         static_cast<long_t>(this->pairs.size()));
                          ++io )
                    {
                        work_type::execute( i
                                            + this->pairs[io].ioffset
                                            + b * INPUT_NEXT_BATCH,
                                            o + this->pairs[io].ooffset
                                            + b * OUTPUT_NEXT_BATCH,
                                            w + this->pairs[io].koffset );
                    }
                }
            };
        }
    }

    void sch( float const * i,
              float const * o,
              float       * w )
    {
        schedule_threads(i, o, w);
    }

    std::vector<upd_problem_args> pairs;

    static const long_t pack_offset =
        CD*CH*CW*OFM_SETS*IFM_SETS*SIMD_WIDTH*SIMD_WIDTH;

public:
    static const long_t workspace_size = pack_offset * 4;

public:
    upd_plan( kernel_launcher * l,
              float const * i,
              float const * o,
              float       * k,
              float       * k_flip,
              float       * b,
              float       * w)
        : launcher(l)
        , conv_fns(Threads)
        , reduce_fns(Threads)
        , pairs(pair_split( IFM_SETS, OFM_SETS, 0, 0, 0 ))
    {
        sch(i, o, w);
        schedule_collect(w,k,0.1);

    }

    void execute()
    {
        launcher->launch( &(conv_fns[0]) );
        launcher->launch( &(reduce_fns[0]) );
    }

    long_t flops() const
    {
        return BS * IFM * OFM * OFD * OFH * OFW * CD * CH * CW * 2;
    }

    double gflops() const
    {
        return static_cast<double>(flops()) / 1000000000;
    }
};



}} // namespace znn:phi
