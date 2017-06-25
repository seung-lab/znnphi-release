#pragma once

#include "znn/layer/conv/update/problem.hpp"
#include <tuple>
#include <utility>

namespace znn
{
namespace phi
{
namespace update
{

template <class... Problems>
using problems_t = std::tuple<Problems...>;

template <class...>
struct problem_cat;

template <class... Ts>
using problem_cat_t = typename problem_cat<Ts...>::type;

template <class... Problems>
struct problem_cat<problems_t<Problems...>>
{
    using type = problems_t<Problems...>;
};

template <class... As, class... Bs, class... Cs>
struct problem_cat<problems_t<As...>, problems_t<Bs...>, Cs...>
{
    using type = problem_cat_t<problems_t<As..., Bs...>, Cs...>;
};

} // namespace update
} // namespace phi
} // namespace znn
