#pragma once

#include "znn/intrin.hpp"
#include "znn/types.hpp"

#include <type_traits>

namespace znn
{
namespace phi
{

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<ZERO, SIMD_FLOAT>::type
    conditional_load(float const*)
{
    return SIMD_ZERO();
}

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<!ZERO, SIMD_FLOAT>::type
    conditional_load(float const* address)
{
    return SIMD_LOAD(address);
}

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<ZERO, SIMD_FLOAT>::type
    conditional_load_or_bias(float const*, float const* b)
{
    return SIMD_LOAD(b);
}

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<!ZERO, SIMD_FLOAT>::type
    conditional_load_or_bias(float const* address, float const*)
{
    return SIMD_LOAD(address);
}
}
} // namespace znn:phi
