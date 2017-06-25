#pragma once

#include <type_traits>

namespace znn
{
namespace phi
{

template <class...>
struct conditional_t;

template <class T>
struct conditional_t<T>
{
    using type = T;
};

template <class C, class T, class... Os>
struct conditional_t<C, T, Os...>
{
    using type =
        typename std::conditional<C::value, T,
                                  typename conditional_t<Os...>::type>::type;
};

template <bool C>
struct condition_t : std::integral_constant<bool, C>
{
};

template <class T>
struct type_wrapper_t
{
    using type = T;
};
}
} // namespace znn:phi
