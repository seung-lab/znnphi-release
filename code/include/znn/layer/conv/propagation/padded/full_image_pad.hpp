#pragma once

#include "znn/layer/conv/propagation/blocking.hpp"
#include "znn/layer/conv/propagation/sub_image_pad.hpp"

namespace znn
{
namespace phi
{
namespace propagation
{

template <bool   First,                 // load or set to bias
          long_t IFMs,                  // number of input images
          class ID, class IH, class IW, // image traits
          class CD, class CH, class CW> // convolution traits
struct full_image_pad
{

private:
    static void loop_over_w(float const* __restrict i, float* __restrict o,
                            float const* __restrict k,
                            float const* __restrict b,
                            long_t const* __restrict valid_d,
                            long_t const* __restrict valid_h)
    {
        static constexpr long_t Full    = IW::size / SIMD_W_BLOCK;
        static constexpr long_t Partial = IW::size % SIMD_W_BLOCK;

        for (long_t w = 0; w < Full; ++w)
        {
            sub_image_pad<First, IFMs, ID, IH, IW, CD, CH, CW,
                          SIMD_W_BLOCK>::execute(i, o, k, b, valid_d, valid_h);
            i += SIMD_W_BLOCK * CW::conv_stride * IW::in_stride;
            o += SIMD_W_BLOCK * IW::out_stride;
        }

        if (Partial)
        {
            sub_image_pad<First, IFMs, ID, IH, IW, CD, CH, CW,
                          Partial>::execute(i, o, k, b, valid_d, valid_h);
        }
    }

    static void loop_over_h(float const* __restrict i, float* __restrict o,
                            float const* __restrict k,
                            float const* __restrict b,
                            long_t const* __restrict valid_d,
                            long_t const* __restrict valid_h)
    {
        for (long_t h = 0; h < IH::size; ++h)
        {
            loop_over_w(i, o, k, b, valid_d, valid_h);
            i += CH::conv_stride * IH::in_stride;
            o += IH::out_stride;
            valid_h += 2;
        }
    }

public:
    static long_t flops()
    {
        return IFMs * SIMD_WIDTH * ID::size * IH::size * IW::size * CD::size *
               CH::size * CW::size * 2;
    }

    static void execute(float const* __restrict i, float* __restrict o,
                        float const* __restrict k, float const* __restrict b,
                        long_t const* __restrict valid_d,
                        long_t const* __restrict valid_h)
    {
        for (long_t d = 0; d < ID::size; ++d)
        {
            loop_over_h(i, o, k, b, valid_d, valid_h);
            i += CD::conv_stride * ID::in_stride;
            o += ID::out_stride;
            valid_d += 2;
        }
    }
};

} // namespace propagation
} // namespace phi
} // namespace znn
