#pragma once

#include "znn/layer/conv/propagation/executable.hpp"
#include "znn/layer/conv/propagation/full_layer/problem.hpp"
#include "znn/layer/conv/propagation/full_layer/split.hpp"
#include "znn/layer/conv/propagation/sub_layer.hpp"
#include "znn/util/constexpr.hpp"

#include <iostream>
#include <type_traits>

namespace znn
{
namespace phi
{
namespace propagation
{

template <class P>
struct serial_scheduler
{
private:
    using sub = typename P::sub_problem;
    static_assert(P::threads == 1, "number of threads not 1");

public:
    static void schedule(long_t t, exec_vector& ev)
    {
        ev[t].push_back(&sub_layer<P>::execute);
    }
};

template <class P>
struct scheduler;

template <class P, long_t N>
struct split_scheduler
{
private:
    using splitted = split_t<P, N>;

    template <long_t                  K>
    static std::enable_if_t<(K == N)> schedule_part(long_t       t0, long_t,
                                                    exec_vector& ev)
    {
        scheduler<typename splitted::rest_t>::schedule(t0, ev);
    }

    template <long_t                 K>
    static std::enable_if_t<(K < N)> schedule_part(long_t t0, long_t t,
                                                   exec_vector& ev)
    {
        scheduler<typename splitted::template part_t<K>>::schedule(t, ev);
        split_scheduler::template schedule_part<K + 1>(t0, t + P::threads / N,
                                                       ev);
    }

public:
    static void schedule(long_t t, exec_vector& ev)
    {
        split_scheduler::template schedule_part<0>(t, t, ev);
    }
};

template <class P>
struct scheduler
{
private:
    using type = std::conditional_t<
        P::threads == 1, serial_scheduler<P>,
        split_scheduler<P, smallest_prime_factor(P::threads)>>;

public:
    static void schedule(long_t t, exec_vector& ev) { type::schedule(t, ev); }
};

template <>
struct scheduler<null_problem_t>
{
    static void schedule(long_t, exec_vector&) {}
};

} // namespace propagation
} // namespace phi
} // namespace znn
