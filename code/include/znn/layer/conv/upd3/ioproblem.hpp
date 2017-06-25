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

template <long_t T, long_t I, long_t O, class Strides, long_t IOff = 0,
          long_t OOff = 0, long_t KOff = 0>
struct upd_io_problem_t
{
    static constexpr long_t threads = T;
    static constexpr long_t i       = I;
    static constexpr long_t o       = O;

    using strides = Strides;

    static constexpr long_t ioff = IOff;
    static constexpr long_t ooff = OOff;
    static constexpr long_t koff = KOff;
};

template <long_t IStride, long_t OStride, long_t KIStride, long_t KOStride>
struct upd_io_problem_strides
{
    static constexpr long_t istride  = IStride;
    static constexpr long_t ostride  = OStride;
    static constexpr long_t kistride = KIStride;
    static constexpr long_t kostride = KOStride;
};

// List of problems

template <class... Ts>
using upd_io_problems_t = std::tuple<Ts...>;

// Merging lists of problems

template <class...>
struct upd_io_problem_cat;

template <class... As>
struct upd_io_problem_cat<upd_io_problems_t<As...>>
{
    using type = upd_io_problems_t<As...>;
};

template <class... As, class... Bs, class... Cs>
struct upd_io_problem_cat<upd_io_problems_t<As...>, upd_io_problems_t<Bs...>,
                          Cs...>
{
    using type = typename upd_io_problem_cat<upd_io_problems_t<As..., Bs...>,
                                             Cs...>::type;
};

template <class... Ts>
using upd_io_problem_cat_t = typename upd_io_problem_cat<Ts...>::type;

// Printing io problems

template <class P>
struct upd_io_problem_printer
{
    static void print()
    {
        std::cout << "Threads: " << P::threads << " i: " << P::i
                  << " o: " << P::o << "  offsets: " << ' ' << P::ioff << ' '
                  << P::ooff << ' ' << P::koff << "\n";
    }
};

template <class...>
struct upd_io_problems_printer_h
{
    static void print() {}
};

template <class A>
struct upd_io_problems_printer_h<A>
{
    static void print() { upd_io_problem_printer<A>::print(); }
};

template <class A, class... Rest>
struct upd_io_problems_printer_h<A, Rest...>
{
    static void print()
    {
        upd_io_problem_printer<A>::print();
        upd_io_problems_printer_h<Rest...>::print();
    }
};

template <class>
struct upd_io_problems_printer;

template <class... As>
struct upd_io_problems_printer<upd_io_problems_t<As...>>
{
    static void print() { upd_io_problems_printer_h<As...>::print(); }
};

// Splitting the problem

template <class P, long_t N>
struct upd_io_problem_split_i
{
    static constexpr long_t parts = N;

    template <long_t K>
    struct get
    {
        using type = upd_io_problem_t<
            P::threads / N, P::i / N, P::o, typename P::strides,
            P::ioff + P::strides::istride*(P::i / N) * K, P::ooff,
            P::koff + P::strides::kistride*(P::i / N) * K>;
    };
};

template <class P, long_t N>
struct upd_io_problem_split_o
{
    static constexpr long_t parts = N;

    template <long_t K>
    struct get
    {
        using type =
            upd_io_problem_t<P::threads / N, P::i, P::o / N,
                             typename P::strides, P::ioff,
                             P::ooff + P::strides::ostride*(P::o / N) * K,
                             P::koff + P::strides::kostride*(P::o / N) * K>;
    };
};

// Forward declaration

template <class>
struct upd_io_split_problem;

template <class P>
using upd_io_split_problem_t = typename upd_io_split_problem<P>::type;

template <class, class>
struct upd_io_recursive_split_h;

template <class C, long_t... I>
struct upd_io_recursive_split_h<C, std::integer_sequence<long_t, I...>>
{
    using type = upd_io_problem_cat_t<typename upd_io_split_problem<
        typename C::template get<I>::type>::type...>;
};

template <class C>
struct upd_io_recursive_split
{
    using type = typename upd_io_recursive_split_h<
        C, std::make_integer_sequence<long_t, C::parts>>::type;
};

template <class P>
struct upd_io_split_problem
{
private:
    static constexpr long_t prime = smallest_prime_factor(P::threads);

public:
    static_assert(prime >= 1, "");

    using type = typename conditional_t<
        condition_t<prime == 1>, type_wrapper_t<upd_io_problems_t<P>>,
        // condition_t< (P::i > P::o) && (P::i % prime == 0) >,
        // upd_io_recursive_split< upd_io_problem_split_i<P, prime>>,
        condition_t<P::o % prime == 0>,
        upd_io_recursive_split<upd_io_problem_split_o<P, prime>>,
        condition_t<P::i % prime == 0>,
        upd_io_recursive_split<upd_io_problem_split_i<P, prime>>,
        type_wrapper_t<upd_io_problems_t<P>>>::type::type;
};

template <class P>
using upd_io_split_problem_t = typename upd_io_split_problem<P>::type;
}
} // namespace znn:phi
