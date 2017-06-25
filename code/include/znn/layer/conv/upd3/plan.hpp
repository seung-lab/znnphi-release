#pragma once

#include "znn/layer/conv/upd3/problem.hpp"
#include "znn/layer/conv/upd3/ioproblem.hpp"
#include "znn/layer/conv/upd3/work.hpp"
#include "znn/util/kernel_launcher.hpp"
#include "znn/layer/dimension.hpp"

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
    kernel_launcher * launcher;
    std::vector<std::function<void()>> conv_fns;

private:
    static constexpr long_t OFM_SETS = (OFM+SIMD_WIDTH-1)/SIMD_WIDTH;
    static constexpr long_t IFM_SETS = (IFM+SIMD_WIDTH-1)/SIMD_WIDTH;

    static constexpr long_t OFM_STRIDE = OFD * OFH * OFW * SIMD_WIDTH;
    static constexpr long_t IFM_STRIDE = IFD * IFH * IFW * SIMD_WIDTH;

    static constexpr long_t IB_STRIDE = IFM_STRIDE * IFM_SETS;
    static constexpr long_t OB_STRIDE = OFM_STRIDE * OFM_SETS;

    static constexpr long_t W_IN_STRIDE  = CD * CH * CW * SIMD_WIDTH;
    static constexpr long_t W_OUT_STRIDE = W_IN_STRIDE * IFM_SETS;

    static constexpr long_t I_STRIDE = IFM_STRIDE * IFM_SETS;
    static constexpr long_t O_STRIDE = OFM_STRIDE * OFM_SETS;

    static const long_t K_STRIDE = W_OUT_STRIDE * OFM_SETS; // for workspace

    using io_strides = upd_io_problem_strides< IFM_STRIDE, OFM_STRIDE,
                                               W_IN_STRIDE, W_OUT_STRIDE >;

    using ishape = upd_ioshape_t< IFM_STRIDE * IFM_SETS,
                                  IFH * IFW * SIMD_WIDTH,
                                  IFW * SIMD_WIDTH >;

    using oshape = upd_ioshape_t< OFM_STRIDE * OFM_SETS,
                                  OFH * OFW * SIMD_WIDTH,
                                  OFW * SIMD_WIDTH >;
    // IO SPLIT PROBLEM
    using io_problem = upd_io_problem_t< Threads,
                                         IFM_SETS,
                                         OFM_SETS,
                                         io_strides >;

    using io_problems = upd_io_split_problem_t<io_problem>;

    using single_io_problem =
        typename std::tuple_element<0,io_problems>::type;

    using problem = upd_problem_t<
        single_io_problem::threads,
        upd_problem_size_t< BS, OFD, OFH >,
        ishape, oshape >;

    using problems = upd_split_problem_t<problem>;

    static const long_t KERNEL_COPIES = single_io_problem::threads;
    static const long_t pack_offset =
        CD*CH*CW*OFM_SETS*IFM_SETS*SIMD_WIDTH*SIMD_WIDTH;

public:
    static const long_t workspace_size = pack_offset * (KERNEL_COPIES) * 4;

private:
    template< class P, long_t NIN, long_t NOUT >
    typename std::enable_if<NIN==0||NOUT==0>::type
    execute_problem_helper( float const *,
                            float const *,
                            float       *) const
    {}

    template< class P, long_t NIN, long_t NOUT >
    typename std::enable_if<NIN==1&&NOUT==1>::type
    execute_problem_helper( float const * a,
                            float const * b,
                            float       * c) const
    {
        P::execute(a,b,c);
    }

    template< class P, long_t NIN, long_t NOUT >
    typename std::enable_if<(NIN>1)&&(NOUT==1)>::type
    execute_problem_helper( float const * a,
                            float const * b,
                            float       * c) const
    {
        upd_plan::template execute_problem_helper<P,NIN/2,NOUT>
            ( a, b, c );
        upd_plan::template execute_problem_helper<P,NIN-NIN/2,NOUT>
            ( a + (NIN/2) * IFM_STRIDE, b, c + (NIN/2) * W_IN_STRIDE);
    }

    template< class P, long_t NIN, long_t NOUT >
    typename std::enable_if<(NOUT>1)&&(NIN>0)>::type
    execute_problem_helper( float const * a,
                            float const * b,
                            float       * c) const
    {
        upd_plan::template execute_problem_helper<P,NIN,NOUT/2>
            ( a, b, c );
        upd_plan::template execute_problem_helper<P,NIN,NOUT-NOUT/2>
            ( a, b + (NOUT/2) * OFM_STRIDE, c + (NOUT/2) * W_OUT_STRIDE);
    }

    // template< class P, long_t NIN, long_t NOUT >
    // typename std::enable_if<(NIN>NOUT)&&(NIN>1)&&(NOUT>0)>::type
    // execute_problem_helper( float const * a,
    //                         float const * b,
    //                         float       * c) const
    // {
    //     upd_plan::template execute_problem_helper<P,NIN/2,NOUT>
    //         ( a, b, c );
    //     upd_plan::template execute_problem_helper<P,NIN-NIN/2,NOUT>
    //         ( a + (NIN/2) * IFM_STRIDE, b, c + (NIN/2) * W_IN_STRIDE);
    // }

    // template< class P, long_t NIN, long_t NOUT >
    // typename std::enable_if<(NIN<=NOUT)&&(NOUT>1)&&(NIN>0)>::type
    // execute_problem_helper( float const * a,
    //                         float const * b,
    //                         float       * c) const
    // {
    //     upd_plan::template execute_problem_helper<P,NIN,NOUT/2>
    //         ( a, b, c );
    //     upd_plan::template execute_problem_helper<P,NIN,NOUT-NOUT/2>
    //         ( a, b + (NOUT/2) * OFM_STRIDE, c + (NOUT/2) * W_OUT_STRIDE);
    // }


private:
    template< long_t N >
    typename std::enable_if< (N >= std::tuple_size<io_problems>::value) >::type
    schedule_io_problem( long_t, float const *, float const *, float * )
    {}

    template< long_t N >
    typename std::enable_if< (N < std::tuple_size<io_problems>::value) >::type
    schedule_io_problem( long_t t, float const * a, float const * b, float * c )
    {
        using iop = typename std::tuple_element<N,io_problems>::type;

        upd_plan::template schedule_problem<0>( t,
                                                a + iop::ioff,
                                                b + iop::ooff,
                                                c + iop::koff );

        upd_plan::template schedule_io_problem<N+1>
            ( t + 1, a, b, c );
    }

    template< long_t N >
    typename std::enable_if< (N >= std::tuple_size<problems>::value) >::type
    schedule_problem( long_t, float const *, float const *, float * )
    {}

    template< long_t N >
    typename std::enable_if< (N < std::tuple_size<problems>::value) >::type
    schedule_problem( long_t t, float const * a, float const * b, float * c )
    {
        using pp = typename std::tuple_element<N,problems>::type;

        using work = upd_work<
            dimension<pp::size::depth,IFH*IFW,OFH*OFW>,
            dimension<pp::size::height,IFW,OFW>,
            dimension<OFW,1,1>,
            conv_traits<CD,1,1>,
            conv_traits<CH,1,1>,
            conv_traits<CW,1,1>>;

        conv_fns[t] = [=]() {
            for ( long_t x = 0; x < pp::size::batch; ++x )
            {
                for ( long_t o = 0; o < single_io_problem::o; ++o )
                {
                    for ( long_t i = 0; i < single_io_problem::i; ++i )
                    {
                        work::execute( a + pp::ioffset + x * IB_STRIDE + i * IFM_STRIDE,
                                       b + pp::ooffset + x * OB_STRIDE + o * OFM_STRIDE,
                                       c + i * W_IN_STRIDE + o * W_OUT_STRIDE );
                    }
                }
                // this->template execute_problem_helper
                // < work, single_io_problem::i, single_io_problem::o>
                // ( a + pp::ioffset + x * IB_STRIDE,
                //   b + pp::ooffset + x * OB_STRIDE,
                //   c );
            }
        };

        std::cout << "Schedulied thread: " << t
                  << ' ' << a << ' ' << b << ' ' << c << "\n";
        std::cout << "\tFLOPS: " << (static_cast<double>(work::flops())/1000000000) *
            pp::size::batch * single_io_problem::i * single_io_problem::o << "\n";

        std::cout << "\tWork: " << pp::size::depth
                  << ' ' << pp::size::height
                  << ' ' << OFW << "\n";


        upd_plan::template schedule_problem<N+1>
            ( t + Threads/single_io_problem::threads, a, b, c + pack_offset );
    }


public:
    upd_plan( kernel_launcher * l,
              float const * i,
              float const * o,
              float       * ,
              float       * ,
              float       * ,
              float       * w)
        : launcher(l)
        , conv_fns(Threads)
    {
        // PRINT_CONST(std::tuple_size<io_problems>::value);
        upd_io_problems_printer<io_problems>::print();
        upd_problems_printer<problems>::print();

        // float * x;

        // //execute_problem_helper<void, 3, 4>( x,x,x );

        upd_plan::template schedule_io_problem<0>(0, i, o, w);

    }

    long_t flops() const
    {
        return BS * IFM * OFM * OFD * OFH * OFW * CD * CH * CW * 2;
    }

    double gflops() const
    {
        return static_cast<double>(flops()) / 1000000000;
    }

    void execute()
    {
        launcher->launch( &(conv_fns[0]) );
        //launcher->launch( &(reduce_fns[0]) );
    }

};




}} // namespace znn:phi
