#pragma once

#include "znn/layer/conv/upd3/ioproblem.hpp"
#include "znn/layer/conv/upd3/problem.hpp"
#include "znn/layer/conv/upd3/work.hpp"
#include "znn/layer/dimension.hpp"
#include "znn/util/kernel_launcher.hpp"

#define PRINT_CONST(c) std::cout << #c << ": " << c << std::endl

namespace znn
{
namespace phi
{

template <long_t Threads, long_t BS, long_t IFM, long_t OFM, long_t IFD,
          long_t IFH, long_t IFW, long_t OFD, long_t OFH, long_t OFW, long_t CD,
          long_t CH, long_t CW>
class upd_plan
{
private:
    kernel_launcher*                   launcher;
    std::vector<std::function<void()>> conv_fns;

private:
    static const long_t OFM_SETS = (OFM + SIMD_WIDTH - 1) / SIMD_WIDTH;
    static const long_t IFM_SETS = (IFM + SIMD_WIDTH - 1) / SIMD_WIDTH;

    static const long_t OFM_STRIDE = OFD * OFH * OFW * SIMD_WIDTH;
    static const long_t IFM_STRIDE = IFD * IFH * IFW * SIMD_WIDTH;

    static const long_t IB_STRIDE = IFM_STRIDE * IFM_SETS;
    static const long_t OB_STRIDE = OFM_STRIDE * OFM_SETS;

    static const long_t W_IN_STRIDE  = CD * CH * CW * SIMD_WIDTH;
    static const long_t W_OUT_STRIDE = W_IN_STRIDE * IFM_SETS;

    static const long_t I_STRIDE = IFM_STRIDE * IFM_SETS;
    static const long_t O_STRIDE = OFM_STRIDE * OFM_SETS;

    static const long_t K_STRIDE = W_OUT_STRIDE * OFM_SETS; // for workspace

    using ishape = upd_ioshape_t<IFM_STRIDE * IFM_SETS, IFH * IFW * SIMD_WIDTH,
                                 IFW * SIMD_WIDTH>;

    using oshape = upd_ioshape_t<OFM_STRIDE * OFM_SETS, OFH * OFW * SIMD_WIDTH,
                                 OFW * SIMD_WIDTH>;

    static const long_t IO_PAIR_GROUPS = gcd(Threads, IFM_SETS* OFM_SETS);

    // How many pairs in an I-O group
    static const long_t PAIRS_PER_GROUP =
        (IFM_SETS * OFM_SETS) / IO_PAIR_GROUPS;

    // How many available threads per I-O group
    // If this is greater than one, we need to do the reduction
    static const long_t THREADS_PER_PAIR_GROUP = Threads / IO_PAIR_GROUPS;

    static const long_t NUM_TYPES = THREADS_PER_PAIR_GROUP;

    // Number of threads that execute one type (on same input)
    static const long_t THREADS_PER_TYPE = Threads / NUM_TYPES;

    using problem = upd_problem_t<NUM_TYPES, upd_problem_size_t<BS, OFD, OFH>,
                                  ishape, oshape>;

    using problems = upd_split_problem_t<problem>;

    static const long_t KERNEL_COPIES = NUM_TYPES;

    static const long_t pack_offset =
        CD * CH * CW * OFM_SETS * IFM_SETS * SIMD_WIDTH * SIMD_WIDTH;

public:
    static const long_t workspace_size = pack_offset * (KERNEL_COPIES)*4;

private:
    std::vector<upd_problem_args> pair_split(long_t ifm, long_t ofm,
                                             long_t ioff, long_t ooff,
                                             long_t koff) const
    {
        if ((ofm % 2) == 0)
        {
            std::vector<upd_problem_args> ret =
                pair_split(ifm, ofm / 2, ioff, ooff, koff);

            std::vector<upd_problem_args> rest =
                pair_split(ifm, ofm / 2, ioff, ooff + (ofm / 2) * OFM_STRIDE,
                           koff + (ofm / 2) * W_OUT_STRIDE);

            for (const auto& e : rest)
                ret.push_back(e);

            return ret;
        }
        else if ((ifm % 2) == 0)
        {
            std::vector<upd_problem_args> ret =
                pair_split(ifm / 2, ofm, ioff, ooff, koff);

            std::vector<upd_problem_args> rest =
                pair_split(ifm / 2, ofm, ioff + (ifm / 2) * IFM_STRIDE, ooff,
                           koff + (ifm / 2) * W_IN_STRIDE);

            for (const auto& e : rest)
                ret.push_back(e);

            return ret;
        }
        else
        {
            std::vector<upd_problem_args> ret;

            for (long_t o = 0; o < ofm; ++o)
            {
                for (long_t i = 0; i < ifm; ++i)
                {
                    ret.push_back(upd_problem_args(
                        ioff + i * IFM_STRIDE, ooff + o * OFM_STRIDE,
                        koff + i * W_IN_STRIDE + o * W_OUT_STRIDE));
                }
            }

            return ret;
        }
    }

private:
    template <long_t N>
    typename std::enable_if<(N >= std::tuple_size<problems>::value)>::type
    schedule_type(long_t, float const*, float const*, float*)
    {
    }

    template <long_t N>
    typename std::enable_if<(N < std::tuple_size<problems>::value)>::type
    schedule_type(long_t tno, float const* i, float const* o, float* w)
    {
        using type = typename std::tuple_element<N, problems>::type;

        using work =
            upd_work<dimension<type::size::depth, IFH * IFW, OFH * OFW>,
                     dimension<type::size::height, IFW, OFW>,
                     dimension<OFW, 1, 1>, conv_traits<CD, 1, 1>,
                     conv_traits<CH, 1, 1>, conv_traits<CW, 1, 1>>;

        std::cout << "\tType: " << type::size::batch << ' ' << type::size::depth
                  << ' ' << type::size::height << ' ' << OFW << ' '
                  << type::ioffset << ' ' << type::ooffset << "\n";

        for (long_t x = 0; x < IO_PAIR_GROUPS; ++x)
        {
            std::cout << "\t\tScheduling pairs " << x * PAIRS_PER_GROUP
                      << " to " << (x * PAIRS_PER_GROUP + PAIRS_PER_GROUP - 1)
                      << " on thread " << tno + x << "\n";

            conv_fns[tno + x] = [this, i, o, w, x]() {
                for (long_t b = 0; b < type::size::batch; ++b)
                {
                    for (long_t io = 0; io < PAIRS_PER_GROUP; ++io)
                    {
                        work::execute(
                            i + this->pairs[x * PAIRS_PER_GROUP + io].ioffset +
                                b * IB_STRIDE + type::ioffset,
                            o + this->pairs[x * PAIRS_PER_GROUP + io].ooffset +
                                b * OB_STRIDE + type::ooffset,
                            w + this->pairs[x * PAIRS_PER_GROUP + io].koffset);
                    }
                }
            };
        }

        upd_plan::template schedule_type<N + 1>(tno + THREADS_PER_TYPE, i, o,
                                                w + pack_offset);
    }

    std::vector<upd_problem_args> pairs;

public:
    upd_plan(kernel_launcher* l, float const* i, float const* o, float*, float*,
             float*, float* w)
        : launcher(l)
        , conv_fns(Threads)
        , pairs(pair_split(IFM_SETS, OFM_SETS, 0, 0, 0))
    {
        PRINT_CONST(IO_PAIR_GROUPS);
        PRINT_CONST(PAIRS_PER_GROUP);
        PRINT_CONST(THREADS_PER_PAIR_GROUP);
        PRINT_CONST(NUM_TYPES);
        PRINT_CONST(THREADS_PER_TYPE);
        PRINT_CONST(pairs.size());
        // upd_io_problems_printer<io_problems>::print();
        upd_problems_printer<problems>::print();

        // float * x;

        // //execute_problem_helper<void, 3, 4>( x,x,x );

        upd_plan::template schedule_type<0>(0, i, o, w);
    }

    long_t flops() const
    {
        return BS * IFM * OFM * OFD * OFH * OFW * CD * CH * CW * 2;
    }

    double gflops() const { return static_cast<double>(flops()) / 1000000000; }

    void execute()
    {
        launcher->launch(&(conv_fns[0]));
        // launcher->launch( &(reduce_fns[0]) );
    }
};
}
} // namespace znn:phi
