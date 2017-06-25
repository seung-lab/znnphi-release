#pragma once

#include "znn/layer/conv/fwd4/blocking.hpp"
#include "znn/layer/conv/fwd4/sub_image.hpp"

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
struct full_image
{
private:
    using RB = fwd_blocking_t<ID::size, IH::size, IW::size>;

private:
    template <long_t D, long_t H>
    static void loop_over_w(float const* __restrict i, float* __restrict o,
                            float const* __restrict k,
                            float const* __restrict b)
    {
        static constexpr long_t Full    = IW::size / RB::width;
        static constexpr long_t Partial = IW::size % RB::width;

        for (long_t w = 0; w < Full; ++w)
        {
            sub_image<First, IFMs, ID, IH, IW, CD, CH, CW, D, H,
                      RB::width>::execute(i, o, k, b);
            i += RB::width * CW::conv_stride * IW::in_stride;
            o += RB::width * IW::out_stride;
        }

        if (Partial)
        {
            sub_image<First, IFMs, ID, IH, IW, CD, CH, CW, D, H,
                      Partial>::execute(i, o, k, b);
        }
    }

    template <long_t D, long_t ClangFormatIgnore = 1>
    static void loop_over_h(float const* __restrict i, float* __restrict o,
                            float const* __restrict k,
                            float const* __restrict b)
    {
        static constexpr long_t Full    = IH::size / RB::height;
        static constexpr long_t Partial = IH::size % RB::height;

        for (long_t h = 0; h < Full; ++h)
        {
            full_image::template loop_over_w<D, RB::height>(i, o, k, b);
            i += RB::height * CH::conv_stride * IH::in_stride;
            o += RB::height * IH::out_stride;
        }

        if (Partial)
        {
            full_image::template loop_over_w<D, Partial>(i, o, k, b);
        }
    }

public:
    static long_t flops()
    {
        return IFMs * SIMD_WIDTH * ID::size * IH::size * IW::size * CD::size *
               CH::size * CW::size * 2;
    }

    static void execute(float const* __restrict i, float* __restrict o,
                        float const* __restrict k, float const* __restrict b)
    {
        static constexpr long_t Full    = ID::size / RB::depth;
        static constexpr long_t Partial = ID::size % RB::depth;

        for (long_t d = 0; d < Full; ++d)
        {
            full_image::template loop_over_h<RB::depth>(i, o, k, b);
            i += RB::depth * CD::conv_stride * ID::in_stride;
            o += RB::depth * ID::out_stride;
        }

        if (Partial)
        {
            full_image::template loop_over_h<Partial>(i, o, k, b);
        }
    }
};

} // namespace propagation
} // namespace phi
} // namespace znn
