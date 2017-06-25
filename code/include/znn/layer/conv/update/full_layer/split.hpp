#pragma once

#include "znn/layer/conv/update/full_layer/problems.hpp"
#include "znn/layer/conv/update/problem.hpp"
#include "znn/util/constexpr.hpp"

namespace znn
{
namespace phi
{
namespace update
{

template <class P, long_t N>
struct batch_split_t
{
private:
    using sub = typename P::sub_problem;
    static_assert(sub::b_len >= N, "batch too small");
    static_assert(P::threads % N == 0, "threads not divisible by N");
    static constexpr long_t len        = sub::b_len / N;
    static constexpr long_t full       = sub::b_len % N;
    static constexpr long_t full_start = sub::b_from + (len + 1) * full;

public:
    static constexpr long_t parts = N;

    template <long_t K>
    struct part
    {
        using type = problem_t<
            P::threads / N, typename P::original_problem,
            sub_problem_t<(K < full) ? sub::b_from + (len + 1) * K
                                     : full_start + len*(K - full),
                          (K < full) ? len + 1 : len, sub::d_from, sub::d_len,
                          sub::h_from, sub::h_len, sub::w_from, sub::w_len>>;
    };

    template <long_t K>
    using part_t = typename part<K>::type;
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
    static constexpr long_t parts = N;

    template <long_t K>
    struct part
    {
        using type =
            problem_t<P::threads / N, typename P::original_problem,
                      sub_problem_t<sub::b_from, sub::b_len,
                                    (K < full) ? sub::d_from + (len + 1) * K
                                               : full_start + len*(K - full),
                                    (K < full) ? len + 1 : len, sub::h_from,
                                    sub::h_len, sub::w_from, sub::w_len>>;
    };

    template <long_t K>
    using part_t = typename part<K>::type;
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
    static constexpr long_t parts = N;

    template <long_t K>
    struct part
    {
        using type = problem_t<
            P::threads / N, typename P::original_problem,
            sub_problem_t<sub::b_from, sub::b_len, sub::d_from, sub::d_len,
                          (K < full) ? sub::h_from + (len + 1) * K
                                     : full_start + len*(K - full),
                          (K < full) ? len + 1 : len, sub::w_from, sub::w_len>>;
    };

    template <long_t K>
    using part_t = typename part<K>::type;
};

template <class P, long_t N>
struct dummy_split_t
{
    static constexpr long_t parts = N;

    template <long_t K>
    struct part
    {
        using type = typename std::conditional<
            K == 0,
            problem_t<1, typename P::original_problem, typename P::sub_problem>,
            null_problem_t>::type;
    };

    template <long_t K>
    using part_t = typename part<K>::type;
};

template <class>
struct split_problem;

template <class T>
using split_problem_t = typename split_problem<T>::type;

namespace detail
{

template <class, class>
struct extractor;

template <class C, long_t... I>
struct extractor<C, std::integer_sequence<long_t, I...>>
{
    using type =
        problem_cat_t<split_problem_t<typename C::template part<I>::type>...>;
};

template <class A, class B>
using extractor_t = typename extractor<A, B>::type;

template <class T>
struct wrap_t
{
    using type = T;
};

} // namespace detail

template <class C>
struct recursive_split
{
    using type =
        detail::extractor_t<C, std::make_integer_sequence<long_t, C::parts>>;
};

template <>
struct split_problem<null_problem_t>
{
    using type = problems_t<null_problem_t>;
};

template <class P>
struct split_problem
{
private:
    static constexpr long_t prime = smallest_prime_factor(P::threads);
    using sub                     = typename P::sub_problem;

public:
    using type = typename std::conditional_t<
        prime == 1, detail::wrap_t<problems_t<P>>,
        std::conditional_t<
            sub::b_len % prime == 0, recursive_split<batch_split_t<P, prime>>,
            std::conditional_t<
                sub::d_len % prime == 0,
                recursive_split<depth_split_t<P, prime>>,
                std::conditional_t<
                    sub::h_len % prime == 0,
                    recursive_split<height_split_t<P, prime>>,
                    std::conditional_t<
                        (sub::b_len >= prime),
                        recursive_split<batch_split_t<P, prime>>,
                        std::conditional_t<
                            (sub::d_len >= prime),
                            recursive_split<depth_split_t<P, prime>>,
                            std::conditional_t<
                                (sub::h_len >= prime),
                                recursive_split<height_split_t<P, prime>>,
                                recursive_split<
                                    dummy_split_t<P, P::threads>>>>>>>>>::type;
};

} // namespace update
} // namespace phi
} // namespace znn
