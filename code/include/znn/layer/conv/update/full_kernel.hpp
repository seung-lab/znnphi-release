#pragma once

#include "znn/layer/conv/update/blocking.hpp"
#include "znn/layer/conv/update/sub_kernel.hpp"

namespace znn
{
namespace phi
{
namespace update
{

template <class ID, class IH, class IW, // image traits
          class CD, class CH, class CW> // convolution traits
struct full_kernel
{
private:
    using RB       = update_blocking<CD::size, CH::size, CW::size>;
    using sub_task = sub_kernel<ID, IH, IW, CD, CH, CW, IW::size, RB::s, RB::d,
                                RB::h, RB::w>;

private:
    static void kernel_loop(float const* __restrict i,
                            float const* __restrict o, float* __restrict k)
    {
        for (long_t d = 0; d < CD::size / RB::d; ++d)
        {
            for (long_t h = 0; h < CH::size / RB::h; ++h)
            {
                for (long_t w = 0; w < CW::size / RB::w; ++w)
                {
                    sub_task::execute(i + d * RB::d * ID::in_stride +
                                          h * RB::h * IH::in_stride +
                                          w * RB::w * IW::in_stride,
                                      o, k +
                                             (d * RB::d * CD::ker_stride +
                                              h * RB::h * CH::ker_stride +
                                              w * RB::w * CW::ker_stride) *
                                                 SIMD_WIDTH * SIMD_WIDTH);
                }
            }
        }
    }

public:
    static long_t flops()
    {
        return SIMD_WIDTH * SIMD_WIDTH * ID::size * IH::size * IW::size *
               CD::size * CH::size * CW::size * 2;
    }

    static void execute(float const* __restrict i, float const* __restrict o,
                        float* __restrict k)
    {
        for (long_t d = 0; d < ID::size; ++d)
        {
            for (long_t h = 0; h < IH::size; ++h)
            {
                kernel_loop(i + d * ID::in_stride + h * IH::in_stride,
                            o + d * ID::out_stride + h * IH::out_stride, k);
            }
        }
    }
};

} // namespace update
} // namespace phi
} // namespace znn
