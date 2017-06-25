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
          long_t IFM, long_t OFM,
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

    static const long_t ABS_MAX_PARALLEL_PAIRSX =
        upd_utils::largest_pow2_factor( IFM_SETS * OFM_SETS );

    static const long_t ABS_MAX_PARALLEL_PAIRS =
        (ABS_MAX_PARALLEL_PAIRSX > 0) ? ABS_MAX_PARALLEL_PAIRSX : 1;


    // Maximal parallelization of the I-O pairs
    static const long_t MAX_PARALLEL_PAIRS =
        ( ABS_MAX_PARALLEL_PAIRS >= Threads ) ? Threads : ABS_MAX_PARALLEL_PAIRS;

    // How many groups of I-O pairs are there (to run in parallel)
    static const long_t IO_PAIR_GROUPS =
        (MAX_PARALLEL_PAIRS >= Threads) ? Threads : MAX_PARALLEL_PAIRS;

    // How many pairs in an I-O group
    static const long_t PAIRS_PER_GROUP =
        (IFM_SETS*OFM_SETS) / IO_PAIR_GROUPS;

    // How many available threads per I-O group
    // If this is greater than one, we need to do the reduction
    static const long_t THREADS_PER_PAIR_GROUP =
        Threads / IO_PAIR_GROUPS;

    // How many batch groups
    static const long_t BATCH_GROUPS =
        (THREADS_PER_PAIR_GROUP>=BS) ? BS : THREADS_PER_PAIR_GROUP;


    // How many batches per group
    static const long_t BATCH_PER_GROUP =
        BS / BATCH_GROUPS;

    // How many threads per batch group
    // If this is greather than one, we need to reduce pairs
    // This is done by generating static types for each one
    static const long_t THREADS_PER_BATCH_GROUP =
        THREADS_PER_PAIR_GROUP / BATCH_GROUPS;

    // Number of different types (same as number of threads per batch group)
    static const long_t NUM_TYPES = THREADS_PER_BATCH_GROUP;

    // static_assert( NUM_TYPES ==
    //                upd_utils::div_factor(Threads,BS,IFM_SETS,OFM_SETS),
    //                "Number of types not correct?" );


    // Number of threads for a single type batch group
    static const long_t THREADS_PER_SINGLE_TYPE_BATCH_GROUP =
        BATCH_GROUPS * IO_PAIR_GROUPS;

    // Number of threads that execute one type (on same input)
    static const long_t THREADS_PER_TYPE = Threads / NUM_TYPES;

    // Required number of kernel copies for compute
    static const long_t KERNEL_COPIES = BATCH_GROUPS * NUM_TYPES;

    // Kernel elements per pair
    static const long_t KERNEL_PAIR_ELEMENTS =
        CD * CH * CW * SIMD_WIDTH * SIMD_WIDTH;

    // All kernel strides
    static const long_t KERNEL_NEXT_INPUT       = KERNEL_PAIR_ELEMENTS;
    static const long_t KERNEL_NEXT_OUTPUT      = KERNEL_NEXT_INPUT * IFM_SETS;
    static const long_t KERNEL_NEXT_BATCH_GROUP = KERNEL_NEXT_OUTPUT * OFM_SETS;
    static const long_t KERNEL_NEXT_TYPE        = KERNEL_NEXT_BATCH_GROUP * BATCH_GROUPS;


    static const long_t INPUT_NEXT_BATCH        = IFM_SETS * SIMD_WIDTH * IFD * IFH * IFD;
    static const long_t INPUT_NEXT_BATCH_GROUP  = INPUT_NEXT_BATCH * BATCH_PER_GROUP;

    static const long_t OUTPUT_NEXT_BATCH        = OFM_SETS * SIMD_WIDTH * OFD * OFH * OFD;
    static const long_t OUTPUT_NEXT_BATCH_GROUP  = OUTPUT_NEXT_BATCH * BATCH_PER_GROUP;


    //static const long_t P_PER_THREAD =


    void collect_kernel( float const * __restrict w,
                         float       * __restrict k,
                         float rate,
                         long_t from,
                         long_t to )
    {
        SIMD_FLOAT eta = SIMD_SET1(rate);

        for ( long_t x = 0; x < KERNEL_COPIES; ++x )
        {
            for (long_t y = from; y < to; ++y )
            {
                SIMD_FLOAT a = SIMD_LOAD(k + y * SIMD_WIDTH);
                SIMD_FLOAT b = SIMD_LOAD(w + y * SIMD_WIDTH);
                a = SIMD_FMADD(b,eta,a);
                SIMD_STORE(k + y * SIMD_WIDTH, a);
            }
            w += KERNEL_NEXT_BATCH_GROUP;
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
            // std::cout << "Thread[ " << t << " ] FROM: " << t * k_per_thread
            //           << " TO " << std::min( (t+1) * k_per_thread, total_kernel_elements)
            //           << "\n";
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
                                                    + o * W_OUT_STRIDE ));
                }
            }

            return ret;
        }
    }

    kernel_launcher * launcher;

    std::vector<std::function<void()>> conv_fns;
    std::vector<std::function<void()>> reduce_fns;

    using pair_problem_t = upd_problem_t< upd_problem_size_t< OFD, OFH, OFW >,
                                          upd_ioshape_t< IFH * IFW * SIMD_WIDTH,
                                                         IFW * SIMD_WIDTH,
                                                         SIMD_WIDTH >,
                                          upd_ioshape_t< OFH * OFW * SIMD_WIDTH,
                                                         OFW * SIMD_WIDTH,
                                                         SIMD_WIDTH > >;

    using problems_t = typename upd_divide_problem_t< THREADS_PER_BATCH_GROUP,
                                                      pair_problem_t >::type;

    static const long_t pack_offset =
        CD*CH*CW*OFM_SETS*IFM_SETS*SIMD_WIDTH*SIMD_WIDTH;


private:


    template< long_t TYPE_NO >
    void schedule_batch_group( long_t bno, long_t tno,
                               float const * i,
                               float const * o,
                               float *       w )
    {
        std::cout << "\tScheduling batch " << bno
                  << " to " << ( bno + BATCH_PER_GROUP - 1 )
                  << " on threads " << tno
                  << " to " << ( tno + IO_PAIR_GROUPS - 1 )
                  << "\n";

        using problem_type =
            typename std::tuple_element<TYPE_NO,problems_t>::type;

        std::cout << "\tType: " << BATCH_PER_GROUP
                  << ' ' << problem_type::size::depth
                  << ' ' << problem_type::size::height
                  << ' ' << problem_type::size::width
                  << ' ' << problem_type::ioffset
                  << ' ' << problem_type::ooffset
                  << "\n";


        for ( long_t x = 0; x < IO_PAIR_GROUPS; ++x )
        {
            std::cout << "\t\tScheduling pairs " << x * PAIRS_PER_GROUP
                      << " to " << ( x * PAIRS_PER_GROUP + PAIRS_PER_GROUP - 1 )
                      << " on thread " << tno + x
                      << "\n";

                std::cout << "\tWork: " << problem_type::size::depth
                << ' ' << problem_type::size::height
                << ' ' << problem_type::size::width << "\n";

            conv_fns[tno + x] = [this,i,o,w,bno,x]() {

                using work_type = upd_work<
                dimension<problem_type::size::depth,IFH*IFW,OFH*OFW>,
                dimension<problem_type::size::height,IFW,OFW>,
                dimension<problem_type::size::width,1,1>,
                conv_traits<CD,1,1>,
                conv_traits<CH,1,1>,
                conv_traits<CW,1,1>>;

                for ( long_t b = 0; b < BATCH_PER_GROUP; ++b )
                {
                    for ( long_t io = 0; io < PAIRS_PER_GROUP; ++io )
                    {
                        work_type::execute( i
                                            + this->pairs[x * PAIRS_PER_GROUP + io].ioffset + problem_type::ioffset
                                            + b * INPUT_NEXT_BATCH + bno * INPUT_NEXT_BATCH_GROUP,
                                            o + this->pairs[x * PAIRS_PER_GROUP + io].ooffset + problem_type::ooffset
                                            + b * OUTPUT_NEXT_BATCH + bno * OUTPUT_NEXT_BATCH_GROUP,
                                            w + this->pairs[x * PAIRS_PER_GROUP + io].koffset );
                    }
                }
            };
        }



    }

    template< long_t TYPE_NO >
    typename std::enable_if<(TYPE_NO==NUM_TYPES),void>::type
    schedule_type( long_t, float const *, float const *, float * )
    {
    }

    template< long_t TYPE_NO >
    typename std::enable_if<(TYPE_NO<NUM_TYPES),void>::type
    schedule_type( long_t tno,
                   float const * i,
                   float const * o,
                   float *       w )
    {
        // std::cout << "Scheduling type " << TYPE_NO
        //           << " (on threads " << tno << " to "
        //           << (tno + THREADS_PER_TYPE - 1) << " ";

        //upd_problem_printer<typename std::tuple_element<TYPE_NO,problems_t>::type>::print();

        for ( long_t b = 0; b < BATCH_GROUPS; ++b )
        {
            schedule_batch_group<TYPE_NO>( b * BATCH_PER_GROUP,
                                           tno + b * IO_PAIR_GROUPS,
                                           i + b * INPUT_NEXT_BATCH_GROUP,
                                           o + b * OUTPUT_NEXT_BATCH_GROUP,
                                           w + b * KERNEL_NEXT_BATCH_GROUP );
        }

        schedule_type<TYPE_NO+1>(tno + THREADS_PER_TYPE,
                                 i,o,w + KERNEL_NEXT_TYPE);
    }

    void sch( float const * i,
              float const * o,
              float       * w )
    {
        schedule_type<0>(0, i, o, w);
    }

    std::vector<upd_problem_args> pairs;

public:
    static const long_t workspace_size = pack_offset * (KERNEL_COPIES) * 4;

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
        // std::cout << "IFM SETS  : " << IFM_SETS << "\n";
        // std::cout << "OFM SETS  : " << OFM_SETS << "\n";

        // PRINT_CONST(MAX_PARALLEL_PAIRS);
        // PRINT_CONST(IO_PAIR_GROUPS);
        // PRINT_CONST(PAIRS_PER_GROUP);
        // PRINT_CONST(THREADS_PER_PAIR_GROUP);
        // PRINT_CONST(BATCH_GROUPS);
        // PRINT_CONST(BATCH_PER_GROUP);
        // PRINT_CONST(THREADS_PER_BATCH_GROUP);
        // PRINT_CONST(NUM_TYPES);
        // PRINT_CONST(THREADS_PER_TYPE);
        // PRINT_CONST(KERNEL_COPIES);

        upd_problems_printer<problems_t>::print();

        // auto xx = pair_split( IFM_SETS, OFM_SETS, 0, 0, 0 );
        // for ( const auto & e: xx )
        // {
        //     std::cout << "E: " << e.ioffset << ' ' << e.ooffset << ' ' << e.koffset << "\n";
        // }

        schedule_collect(w,k,0.1);

        sch(i, o, w);
    }

    void execute()
    {
        launcher->launch( &(conv_fns[0]) );
        //launcher->launch( &(reduce_fns[0]) );
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
