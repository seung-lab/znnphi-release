#pragma once

#include "znn/layer/conv/update/full_layer/problems.hpp"
#include "znn/layer/conv/update/problem.hpp"
#include <iostream>

namespace znn
{
namespace phi
{
namespace update
{

template <class P>
struct problem_printer
{
private:
    using sub = typename P::sub_problem;

public:
    static void print()
    {
        std::cout << "Threads: " << P::threads << " batch: " << sub::b_from
                  << "-" << (sub::b_from + sub::b_len - 1)
                  << " depth: " << sub::d_from << "-"
                  << (sub::d_from + sub::d_len - 1)
                  << " height: " << sub::h_from << "-"
                  << (sub::h_from + sub::h_len - 1) << " width: " << sub::w_from
                  << "-" << (sub::w_from + sub::w_len - 1) << "\n";
    }
};

template <>
struct problem_printer<null_problem_t>
{
    static void print() { std::cout << "NULL PROBLEM\n"; }
};

template <class...>
struct problems_printer_helper
{
    static void print() {}
};

template <class A>
struct problems_printer_helper<A>
{
    static void print() { problem_printer<A>::print(); }
};

template <class A, class... Rest>
struct problems_printer_helper<A, Rest...>
{
    static void print()
    {
        problem_printer<A>::print();
        problems_printer_helper<Rest...>::print();
    }
};

template <class>
struct problems_printer;

template <class... As>
struct problems_printer<problems_t<As...>>
{
    static void print() { problems_printer_helper<As...>::print(); }
};

} // namespace update
} // namespace phi
} // namespace znn
