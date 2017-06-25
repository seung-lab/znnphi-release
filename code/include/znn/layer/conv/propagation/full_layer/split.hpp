#pragma once

#include "znn/layer/conv/propagation/full_layer/problem.hpp"
#include <type_traits>

namespace znn
{
namespace phi
{
namespace propagation
{

// Splitting the problem along batches into N chunks + remainder
template <class P, long_t N>
struct batch_split_t
{
private:
    using sub = typename P::sub_problem;
    static_assert(sub::b_len >= N, "batch too small");
    static_assert(P::threads % N == 0, "threads not divisible by N");
    static constexpr long_t len = sub::b_len / N;

public:
    template <long_t K>
    struct part
    {
        using type = problem_t<
            P::threads / N, typename P::original_problem,
            sub_problem_t<sub::b_from + K * len, len, sub::ofm_from,
                          sub::ofm_len, sub::d_from, sub::d_len, sub::h_from,
                          sub::h_len, sub::w_from, sub::w_len>>;
    };
    template <long_t K>
    using part_t = typename part<K>::type;

    using rest_t = std::conditional_t<
        sub::b_len % N == 0, null_problem_t,
        problem_t<
            P::threads, typename P::original_problem,
            sub_problem_t<sub::b_from + N * len, sub::b_len % N, sub::ofm_from,
                          sub::ofm_len, sub::d_from, sub::d_len, sub::h_from,
                          sub::h_len, sub::w_from, sub::w_len>>>;
};

// Splitting the problem along output feature-map sets
template <class P, long_t N>
struct ofm_split_t
{
private:
    using sub = typename P::sub_problem;
    static_assert(sub::ofm_len >= N, "ofm too small");
    static_assert(P::threads % N == 0, "threads not divisible by N");
    static constexpr long_t len = sub::ofm_len / N;

public:
    template <long_t K>
    struct part
    {
        using type = problem_t<
            P::threads / N, typename P::original_problem,
            sub_problem_t<sub::b_from, sub::b_len, sub::ofm_from + K * len, len,
                          sub::d_from, sub::d_len, sub::h_from, sub::h_len,
                          sub::w_from, sub::w_len>>;
    };
    template <long_t K>
    using part_t = typename part<K>::type;

    using rest_t = std::conditional_t<
        sub::ofm_len % N == 0, null_problem_t,
        problem_t<
            P::threads, typename P::original_problem,
            sub_problem_t<sub::b_from, sub::b_len, sub::ofm_from + N * len,
                          sub::ofm_len % N, sub::d_from, sub::d_len,
                          sub::h_from, sub::h_len, sub::w_from, sub::w_len>>>;
};

template <class P, long_t N>
struct depth_split_t
{
private:
    using sub = typename P::sub_problem;
    static_assert(sub::d_len >= N, "depth too small");
    static_assert(P::threads % N == 0, "threads not divisible by N");
    static constexpr long_t len        = sub::d_len / N;
    static constexpr long_t full       = sub::d_len % N;
    static constexpr long_t full_start = sub::d_from + (len + 1) * full;

public:
    template <long_t K>
    struct part
    {
        using type = problem_t<
            P::threads / N, typename P::original_problem,
            sub_problem_t<sub::b_from, sub::b_len, sub::ofm_from, sub::ofm_len,
                          (K < full) ? sub::d_from + (len + 1) * K
                                     : full_start + len*(K - full),
                          (K < full) ? len + 1 : len, sub::h_from, sub::h_len,
                          sub::w_from, sub::w_len>>;
    };

    template <long_t K>
    using part_t = typename part<K>::type;

    using rest_t = null_problem_t;
};

template <class P, long_t N>
struct height_split_t
{
private:
    using sub = typename P::sub_problem;
    static_assert(sub::h_len >= N, "height too small");
    static_assert(P::threads % N == 0, "threads not divisible by N");
    static constexpr long_t len        = sub::h_len / N;
    static constexpr long_t full       = sub::h_len % N;
    static constexpr long_t full_start = sub::h_from + (len + 1) * full;

public:
    template <long_t K>
    struct part
    {
        using type = problem_t<
            P::threads / N, typename P::original_problem,
            sub_problem_t<sub::b_from, sub::b_len, sub::ofm_from, sub::ofm_len,
                          sub::d_from, sub::d_len,
                          (K < full) ? sub::h_from + (len + 1) * K
                                     : full_start + len*(K - full),
                          (K < full) ? len + 1 : len, sub::w_from, sub::w_len>>;
    };

    template <long_t K>
    using part_t = typename part<K>::type;

    using rest_t = null_problem_t;
};

template <class P, long_t N>
struct width_split_t
{
private:
    using sub = typename P::sub_problem;
    static_assert(sub::w_len >= N, "width too small");
    static_assert(P::threads % N == 0, "threads not divisible by N");
    static constexpr long_t len        = sub::w_len / N;
    static constexpr long_t full       = sub::w_len % N;
    static constexpr long_t full_start = sub::w_from + (len + 1) * full;

public:
    template <long_t K>
    struct part
    {
        using type = problem_t<
            P::threads / N, typename P::original_problem,
            sub_problem_t<sub::b_from, sub::b_len, sub::ofm_from, sub::ofm_len,
                          sub::d_from, sub::d_len, sub::h_from, sub::h_len,
                          (K < full) ? sub::w_from + (len + 1) * K
                                     : full_start + len*(K - full),
                          (K < full) ? len + 1 : len>>;
    };

    template <long_t K>
    using part_t = typename part<K>::type;

    using rest_t = null_problem_t;
};

template <class P, long_t N>
struct nosplit_t
{
    template <long_t K>
    struct part
    {
        using type =
            std::conditional_t<(K > 0), null_problem_t,
                               problem_t<1, typename P::original_problem,
                                         typename P::sub_problem>>;
    };

    template <long_t K>
    using part_t = typename part<K>::type;

    using rest_t = null_problem_t;
};

template <class P, long_t N>
struct split
{
private:
    using sub = typename P::sub_problem;

public:
    using type = std::conditional_t<
        (sub::b_len >= N), batch_split_t<P, N>,
        std::conditional_t<
            (sub::ofm_len >= N), ofm_split_t<P, N>,
            std::conditional_t<
                (sub::d_len >= N), depth_split_t<P, N>,
                std::conditional_t<(sub::h_len >= N), height_split_t<P, N>,
                                   std::conditional_t<(sub::w_len >= 8 * N),
                                                      width_split_t<P, N>,
                                                      nosplit_t<P, N>>>>>>;
};

template <class P, long_t N>
using split_t = typename split<P, N>::type;

} // namespace propagation
} // namespace phi
} // namespace znn
