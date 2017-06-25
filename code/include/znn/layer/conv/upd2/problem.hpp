#pragma once

#include "znn/types.hpp"
#include <tuple>
#include <iostream>

namespace znn { namespace phi {

struct null_upd_problem_t
{
    static const long_t ioffset = 0;
    static const long_t ooffset = 0;
};

template< class ProblemSize,
          class IShape,
          class OShape,
          long_t InputOffset  = 0,
          long_t OutputOffset = 0 >
struct upd_problem_t
{
    using size   = ProblemSize;
    using ishape = IShape;
    using oshape = OShape;

    static const long_t ioffset = InputOffset;
    static const long_t ooffset = OutputOffset;
};

template< long_t D,
          long_t H,
          long_t W >
struct upd_problem_size_t
{
    static const long_t depth    = D;
    static const long_t height   = H;
    static const long_t width    = W;
};


template< long_t D, long_t H, long_t W >
struct upd_ioshape_t
{
    static const long_t depth  = D;
    static const long_t height = H;
    static const long_t width  = W;
};

template< class, class >
struct upd_problems_cat_t;

template< class... As, class... Bs >
struct upd_problems_cat_t< std::tuple<As...>, std::tuple<Bs...> >
{
    using type = std::tuple<As...,Bs...>;
};

template< class Problem >
struct upd_problem_printer
{
    static void print()
    {
        std::cout << "Size: " << Problem::size::depth
                  << ' ' << Problem::size::height
                  << ' ' << Problem::size::width
                  << "   offsets: " << Problem::ioffset
                  << ' ' << Problem::ooffset << "\n";
    }
};

template<>
struct upd_problem_printer<null_upd_problem_t>
{
    static void print()
    {
        std::cout << "NULL PROBLEM\n";
    }
};

template< class... >
struct upd_problems_printer_h
{
    static void print()
    {
    }
};

template< class A >
struct upd_problems_printer_h<A>
{
    static void print()
    {
        upd_problem_printer<A>::print();
    }
};

template< class A, class... Rest >
struct upd_problems_printer_h<A, Rest...>
{
    static void print()
    {
        upd_problem_printer<A>::print();
        upd_problems_printer_h<Rest...>::print();
    }
};


template< class >
struct upd_problems_printer;

template< class... As >
struct upd_problems_printer< std::tuple<As...> >
{
    static void print()
    {
        upd_problems_printer_h<As...>::print();
    }
};

struct upd_problem_args
{
    long_t ioffset;
    long_t ooffset;
    long_t koffset;

    upd_problem_args( long_t i, long_t o, long_t k )
        : ioffset(i)
        , ooffset(o)
        , koffset(k)
    {}
};

}} // namespace znn:phi
