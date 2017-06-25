#pragma once

#include "znn/intrin.hpp"
#include "znn/types.hpp"

namespace znn
{
namespace phi
{

//#define ZNN_BLOCK_W_MAX (SIMD_MAX_BLOCK)

template <long_t D, long_t H, long_t W>
struct fwd_blocking_t
{
private:
    static constexpr long_t pick_best(long_t z, long_t x, long_t w)
    {
        return (w == 1) ? w
                        : (((z + w - 2) / (w - 1) == x) ? pick_best(z, x, w - 1)
                                                        : w);
    }

public:
    static const long_t w =
        pick_best(W, (W + SIMD_W_BLOCK - 1) / SIMD_W_BLOCK, SIMD_W_BLOCK);
    static const long_t width = w;

private:
    static const long_t rem_for_h = SIMD_MAX_BLOCK / w;

public:
    static const long_t h =
        pick_best(H, (H + rem_for_h - 1) / rem_for_h, rem_for_h);
    static const long_t height = h;

private:
    static const long_t rem_for_d = SIMD_MAX_BLOCK / w / h;

public:
    static const long_t d =
        pick_best(D, (D + rem_for_d - 1) / rem_for_d, rem_for_d);
    static const long_t depth = d;
};

template <long_t D, long_t H, long_t W>
using propagation_blocking = fwd_blocking_t<D, H, W>;
}
} // namespace znn:phi
