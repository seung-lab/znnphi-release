#pragma once

#include "znn/types.hpp"

#include <algorithm>
#include <vector>
#include <zi/utility/singleton.hpp>

namespace znn
{
namespace phi
{
namespace propagation
{

template <long_t O, long_t W, long_t P>
struct pad_bounds
{
    std::vector<long_t> data;

    pad_bounds()
        : data(2 * O)
    {
        data[0] = P;
        for (long_t i = 1; i < O; ++i)
        {
            data[i * 2] = std::max(data[i * 2 - 2] - 1, static_cast<long_t>(0));
        }
        data[2 * O - 1] = W - P;
        for (long_t i = O - 2; i >= 0; --i)
        {
            data[i * 2 + 1] = std::min(W, data[i * 2 + 3] + 1);
        }
    }

    static long_t const* bounds;
};

template <long_t O, long_t W, long_t P>
long_t const* pad_bounds<O, W, P>::bounds =
    zi::singleton<pad_bounds<O, W, P>>::instance().data.data();

} // namespace propagation
} // namespace phi
} // namespace znn
