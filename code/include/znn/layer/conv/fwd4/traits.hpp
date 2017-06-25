#pragma once

#include "znn/types.hpp"

namespace znn
{
namespace phi
{
namespace propagation
{

template <long_t Size, long_t InStride, long_t OutStride>
struct image_traits
{
    static constexpr long_t size       = Size;
    static constexpr long_t in_stride  = InStride;
    static constexpr long_t out_stride = OutStride;
};

template <long_t Size, long_t KerStride, long_t ConvStride = 1>
struct conv_traits
{
    static constexpr long_t size        = Size;
    static constexpr long_t ker_stride  = KerStride;
    static constexpr long_t conv_stride = ConvStride;
};

} // namespace propagation
} // namespace phi
} // namespace znn
