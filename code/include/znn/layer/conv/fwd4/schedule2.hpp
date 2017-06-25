#pragma once

//#include "znn/layer/conv/fwd/split.hpp"
//#include "znn/layer/conv/fwd/split2.hpp"
//#include "znn/layer/conv/fwd/split3.hpp"
#include "znn/layer/conv/fwd4/split_n.hpp"
#include "znn/layer/conv/fwd4/execute.hpp"
#include "znn/meta.hpp"
#include "znn/intrin.hpp"
#include <iostream>
#include <vector>
#include <memory>

namespace znn
{
namespace phi
{

using exec_vector = std::vector<std::vector<std::unique_ptr<fwd_execute>>>;

template <class Problem>
struct fwd_schedule_t;

template <class Problem, long_t Next, long_t Factor>
struct fwd_schedule2_t;

template <class Problem>
struct fwd_serial_schedule_t
{
private:
    using size   = typename Problem::size;
    using wshape = typename Problem::shapes::weight;

public:
    static void schedule(long_t t, exec_vector& e)
    {
        using serial_problem = typename extract_serial_problem<Problem>::type;
        auto task            = std::make_unique<fwd_execute_t<serial_problem>>(
            std::integral_constant<long_t, Problem::ioffset>::value,
            std::integral_constant<long_t, Problem::ooffset>::value,
            std::integral_constant<long_t, Problem::koffset>::value,
            std::integral_constant<long_t, Problem::boffset>::value *
                SIMD_WIDTH);
        std::cout << "Thread : " << t << "  Problem: " << Problem::size::batch
                  << ' ' << Problem::size::ifm << ' ' << Problem::size::ofm_sets
                  << ' ' << Problem::size::depth << ' ' << Problem::size::height
                  << ' ' << Problem::size::width << ' '
                  << "  Offsets: " << Problem::ioffset << ' '
                  << Problem::ooffset << ' ' << Problem::koffset << ' ';

        std::cout << "FLOPS: " << task->gflops() << "\n";
        e[t].push_back(std::move(task));
    }
};

template <class Problem, long_t Next, long_t Factor>
struct fwd_parallel_schedule2_t
{
    using split = typename fwd_splitn_problem_t<Problem, 2, Factor>::type;

    static void schedule(long_t n, exec_vector& e)
    {
        fwd_schedule2_t<typename split::template part<0>::type, Next * 2,
                        Factor * 2>::schedule(n, e);
        fwd_schedule2_t<typename split::template part<1>::type, Next * 2,
                        Factor * 2>::schedule(n + Next, e);
        fwd_schedule2_t<typename split::rest, Next, Factor * 2>::schedule(n, e);
    }
};

template <class Problem, long_t Next, long_t Factor>
struct fwd_parallel_schedule3_t
{
    using split = typename fwd_splitn_problem_t<Problem, 3, Factor>::type;

    static void schedule(long_t n, exec_vector& e)
    {
        fwd_schedule2_t<typename split::template part<0>::type, Next * 3,
                        Factor * 3>::schedule(n, e);
        fwd_schedule2_t<typename split::template part<1>::type, Next * 3,
                        Factor * 3>::schedule(n + Next, e);
        fwd_schedule2_t<typename split::template part<2>::type, Next * 3,
                        Factor * 3>::schedule(n + Next + Next, e);
        fwd_schedule2_t<typename split::rest, Next, Factor * 2>::schedule(n, e);
    }
};

template <class Problem, long_t Next, long_t Factor>
struct fwd_parallel_schedule5_t
{
    using split = typename fwd_splitn_problem_t<Problem, 5, Factor>::type;

    static void schedule(long_t n, exec_vector& e)
    {
        fwd_schedule2_t<typename split::template part<0>::type, Next * 5,
                        Factor * 5>::schedule(n, e);
        fwd_schedule2_t<typename split::template part<1>::type, Next * 5,
                        Factor * 5>::schedule(n + Next, e);
        fwd_schedule2_t<typename split::template part<2>::type, Next * 5,
                        Factor * 5>::schedule(n + Next * 2, e);
        fwd_schedule2_t<typename split::template part<3>::type, Next * 5,
                        Factor * 5>::schedule(n + Next * 3, e);
        fwd_schedule2_t<typename split::template part<4>::type, Next * 5,
                        Factor * 5>::schedule(n + Next * 4, e);
        fwd_schedule2_t<typename split::rest, Next, Factor * 2>::schedule(n, e);
    }
};

template <class Problem, long_t Next, long_t Factor>
struct fwd_parallel_schedule7_t
{
    using split = typename fwd_splitn_problem_t<Problem, 7, Factor>::type;

    static void schedule(long_t n, exec_vector& e)
    {
        fwd_schedule2_t<typename split::template part<0>::type, Next * 7,
                        Factor * 7>::schedule(n, e);
        fwd_schedule2_t<typename split::template part<1>::type, Next * 7,
                        Factor * 7>::schedule(n + Next, e);
        fwd_schedule2_t<typename split::template part<2>::type, Next * 7,
                        Factor * 7>::schedule(n + Next * 2, e);
        fwd_schedule2_t<typename split::template part<3>::type, Next * 7,
                        Factor * 7>::schedule(n + Next * 3, e);
        fwd_schedule2_t<typename split::template part<4>::type, Next * 7,
                        Factor * 7>::schedule(n + Next * 4, e);
        fwd_schedule2_t<typename split::template part<5>::type, Next * 7,
                        Factor * 7>::schedule(n + Next * 5, e);
        fwd_schedule2_t<typename split::template part<6>::type, Next * 7,
                        Factor * 7>::schedule(n + Next * 6, e);
        fwd_schedule2_t<typename split::rest, Next, Factor * 2>::schedule(n, e);
    }
};

template <class Problem, long_t Next, long_t Factor>
struct fwd_schedule2_t
{
    using type = typename std::conditional<
        Problem::threads == 1, fwd_serial_schedule_t<Problem>,
        typename std::conditional<
            (Problem::threads % 2) == 0,
            fwd_parallel_schedule2_t<Problem, Next, Factor>,
            typename std::conditional<
                (Problem::threads % 3) == 0,
                fwd_parallel_schedule3_t<Problem, Next, Factor>,
                typename std::conditional<
                    (Problem::threads % 5) == 0,
                    fwd_parallel_schedule5_t<Problem, Next, Factor>,
                    typename std::conditional<
                        (Problem::threads % 7) == 0,
                        fwd_parallel_schedule7_t<Problem, Next, Factor>,
                        int>::type>::type>::type>::type>::type;

    static void schedule(long_t t, exec_vector& e) { type::schedule(t, e); }
};

template <long_t Next, long_t Factor>
struct fwd_schedule2_t<null_fwd_problem_t, Next, Factor>
{
    static void schedule(long_t, exec_vector&){};
};

template <class Problem>
struct fwd_schedule_t
{
    static void schedule(long_t t, exec_vector& e)
    {
        fwd_schedule2_t<Problem, 1, 0>::schedule(t, e);
    }
};

template <>
struct fwd_schedule_t<null_fwd_problem_t>
{
    static void schedule(long_t, exec_vector&){};
};
}
} // namespace znn:phi
