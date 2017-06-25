#pragma once

#include "znn/layer/conv/upd3/utils.hpp"
#include "znn/meta.hpp"
#include "znn/types.hpp"
#include <iostream>
#include <tuple>
#include <utility>

namespace znn
{
namespace phi
{

template <long_t Threads, class ProblemSize, class IShape, class OShape,
          long_t InputOffset = 0, long_t OutputOffset = 0>
struct upd_problem_t
{
    using size   = ProblemSize;
    using ishape = IShape;
    using oshape = OShape;

    static constexpr long_t threads = Threads;
    static constexpr long_t ioffset = InputOffset;
    static constexpr long_t ooffset = OutputOffset;
};

template <long_t B, long_t D, long_t H>
struct upd_problem_size_t
{
    static constexpr long_t batch  = B;
    static constexpr long_t depth  = D;
    static constexpr long_t height = H;
};

struct null_upd_problem_t
{
    static constexpr long_t ioffset = 0;
    static constexpr long_t ooffset = 0;

    using size = upd_problem_size_t<0, 0, 0>;
};

template <long_t B, long_t D, long_t H>
struct upd_ioshape_t
{
    static constexpr long_t batch  = B;
    static constexpr long_t depth  = D;
    static constexpr long_t height = H;
};

template <class... Ts>
using upd_problems_t = std::tuple<Ts...>;

template <class...>
struct upd_problem_cat;

template <class... As>
struct upd_problem_cat<upd_problems_t<As...>>
{
    using type = upd_problems_t<As...>;
};

template <class... As, class... Bs, class... Cs>
struct upd_problem_cat<upd_problems_t<As...>, upd_problems_t<Bs...>, Cs...>
{
    using type =
        typename upd_problem_cat<upd_problems_t<As..., Bs...>, Cs...>::type;
};

template <class... Ts>
using upd_problem_cat_t = typename upd_problem_cat<Ts...>::type;

template <class P>
struct upd_problem_printer
{
    static void print()
    {
        std::cout << "Threads: " << P::threads << " size: " << P::size::batch
                  << ' ' << P::size::depth << ' ' << P::size::height
                  << "  offsets: " << ' ' << P::ioffset << ' ' << P::ooffset
                  << "\n";
    }
};

template <>
struct upd_problem_printer<null_upd_problem_t>
{
    static void print() { std::cout << "NULL PROBLEM\n"; }
};

template <class...>
struct upd_problems_printer_h
{
    static void print() {}
};

template <class A>
struct upd_problems_printer_h<A>
{
    static void print() { upd_problem_printer<A>::print(); }
};

template <class A, class... Rest>
struct upd_problems_printer_h<A, Rest...>
{
    static void print()
    {
        upd_problem_printer<A>::print();
        upd_problems_printer_h<Rest...>::print();
    }
};

template <class>
struct upd_problems_printer;

template <class... As>
struct upd_problems_printer<upd_problems_t<As...>>
{
    static void print() { upd_problems_printer_h<As...>::print(); }
};

template <class>
struct upd_split_problem;

template <class T>
using upd_split_problem_t = typename upd_split_problem<T>::type;

template <class, class>
struct upd_extractor_t;

template <class C, long_t... I>
struct upd_extractor_t<C, std::integer_sequence<long_t, I...>>
{
    using type = upd_problem_cat_t<
        upd_split_problem_t<typename C::template get<I>::type>...>;
};

template <class C>
struct upd_recursive_split_t
{
    using type = typename upd_extractor_t<
        C, std::make_integer_sequence<long_t, C::parts>>::type;
};

template <class P, long_t N>
struct upd_batch_split_t
{
private:
    static constexpr long_t blen = P::size::batch / N;
    static constexpr long_t badd = P::size::batch % N;

    static constexpr long_t bioff =
        P::ioffset + P::ishape::batch * (blen + 1) * badd;

    static constexpr long_t booff =
        P::ooffset + P::oshape::batch * (blen + 1) * badd;

public:
    static constexpr long_t parts = N;

    template <long_t K>
    struct get
    {
        using type = typename std::conditional<
            (K < badd),
            upd_problem_t<
                P::threads / N,
                upd_problem_size_t<blen + 1, P::size::depth, P::size::height>,
                typename P::ishape, typename P::oshape,
                P::ioffset + P::ishape::batch*(blen + 1) * K,
                P::ooffset + P::oshape::batch*(blen + 1) * K>,
            upd_problem_t<
                P::threads / N,
                upd_problem_size_t<blen, P::size::depth, P::size::height>,
                typename P::ishape, typename P::oshape,
                bioff + P::ishape::batch * blen*(K - badd),
                booff + P::oshape::batch * blen*(K - badd)>>::type;
    };
};

template <class P, long_t N>
struct upd_depth_split_t
{
private:
    static constexpr long_t dlen = P::size::depth / N;
    static constexpr long_t dadd = P::size::depth % N;

    static constexpr long_t dioff =
        P::ioffset + P::ishape::depth * (dlen + 1) * dadd;

    static constexpr long_t dooff =
        P::ooffset + P::oshape::depth * (dlen + 1) * dadd;

public:
    static constexpr long_t parts = N;

    template <long_t K>
    struct get
    {
        using type = typename std::conditional<
            (K < dadd),
            upd_problem_t<
                P::threads / N,
                upd_problem_size_t<P::size::batch, dlen + 1, P::size::height>,
                typename P::ishape, typename P::oshape,
                P::ioffset + P::ishape::depth*(dlen + 1) * K,
                P::ooffset + P::oshape::depth*(dlen + 1) * K>,
            upd_problem_t<
                P::threads / N,
                upd_problem_size_t<P::size::batch, dlen, P::size::height>,
                typename P::ishape, typename P::oshape,
                dioff + P::ishape::depth * dlen*(K - dadd),
                dooff + P::oshape::depth * dlen*(K - dadd)>>::type;
    };
};

template <class P, long_t N>
struct upd_height_split_t
{
private:
    static constexpr long_t hlen = P::size::height / N;
    static constexpr long_t hadd = P::size::height % N;

    static constexpr long_t hioff =
        P::ioffset + P::ishape::height * (hlen + 1) * hadd;

    static constexpr long_t hooff =
        P::ooffset + P::oshape::height * (hlen + 1) * hadd;

public:
    static constexpr long_t parts = N;

    template <long_t K>
    struct get
    {
        using type = typename std::conditional<
            (K < hadd),
            upd_problem_t<
                P::threads / N,
                upd_problem_size_t<P::size::batch, P::size::depth, hlen + 1>,
                typename P::ishape, typename P::oshape,
                P::ioffset + P::ishape::height*(hlen + 1) * K,
                P::ooffset + P::oshape::height*(hlen + 1) * K>,
            upd_problem_t<
                P::threads / N,
                upd_problem_size_t<P::size::batch, P::size::depth, hlen>,
                typename P::ishape, typename P::oshape,
                hioff + P::ishape::height * hlen*(K - hadd),
                hooff + P::oshape::height * hlen*(K - hadd)>>::type;
    };
};

template <class P, long_t N>
struct upd_dummy_split_t
{
    static constexpr long_t parts = N;

    template <long_t K>
    struct get
    {
        using type = typename std::conditional<
            K == 0, upd_problem_t<1, typename P::size, typename P::ishape,
                                  typename P::oshape, P::ioffset, P::ooffset>,
            null_upd_problem_t>::type;
    };
};

template <>
struct upd_split_problem<null_upd_problem_t>
{
    using type = upd_problems_t<null_upd_problem_t>;
};

template <class P>
struct upd_split_problem
{
private:
    static constexpr long_t prime = smallest_prime_factor(P::threads);

public:
    static_assert(prime >= 1, "");

    using type = typename conditional_t<
        condition_t<prime == 1>, type_wrapper_t<upd_problems_t<P>>,
        condition_t<P::size::batch % prime == 0>,
        upd_recursive_split_t<upd_batch_split_t<P, prime>>,
        condition_t<P::size::depth % prime == 0>,
        upd_recursive_split_t<upd_depth_split_t<P, prime>>,
        condition_t<P::size::height % prime == 0>,
        upd_recursive_split_t<upd_height_split_t<P, prime>>,
        condition_t<(P::size::batch >= prime)>,
        upd_recursive_split_t<upd_batch_split_t<P, prime>>,
        condition_t<(P::size::depth >= prime)>,
        upd_recursive_split_t<upd_depth_split_t<P, prime>>,
        condition_t<(P::size::height >= prime)>,
        upd_recursive_split_t<upd_height_split_t<P, prime>>,
        upd_recursive_split_t<upd_dummy_split_t<P, P::threads>>>::type::type;
};

template <class P>
using upd_split_problem_t = typename upd_split_problem<P>::type;

struct upd_problem_args
{
    long_t ioffset;
    long_t ooffset;
    long_t koffset;

    upd_problem_args(long_t i, long_t o, long_t k)
        : ioffset(i)
        , ooffset(o)
        , koffset(k)
    {
    }
};
}
} // namespace znn:phi
