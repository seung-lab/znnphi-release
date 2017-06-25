#pragma once

#include "znn/types.hpp"

namespace znn
{
namespace phi
{

inline constexpr long_t smallest_prime_factor(long_t a)
{
    return (a % 2 == 0)
               ? 2
               : ((a % 3 == 0)
                      ? 3
                      : ((a % 5 == 0)
                             ? 5
                             : ((a % 7 == 0)
                                    ? 7
                                    : ((a % 11 == 0)
                                           ? 11
                                           : ((a % 13 == 0) ? 13 : a)))));
}

inline constexpr long_t gcd(long_t a, long_t b)
{
    return (b == 0) ? a : gcd(b, a % b);
}
}
} // namespace znn:phi
