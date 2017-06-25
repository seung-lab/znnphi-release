#pragma once

#include "znn/intrin.hpp"
#include "znn/types.hpp"
#include "znn/util/conditional_load.hpp"

#include <type_traits>

namespace znn
{
namespace phi
{
namespace propagation
{

template <bool   First,                 // load or set to bias
          long_t IFMs,                  // number of input images
          class ID, class IH, class IW, // image traits
          class CD, class CH, class CW, // convolution traits
          long_t RW                     // register blocking
          >
struct sub_image_pad;

struct sub_image_pad_dummy
{
    static void execute(float const* __restrict, float* __restrict,
                        float const* __restrict, float const* __restrict,
                        long_t const* __restrict, long_t const* __restrict)
    {
    }
};

template <bool   First,                 // load or set to bias
          long_t IFMs,                  // number of input images
          class ID, class IH, class IW, // image traits
          class CD, class CH, class CW, // convolution traits
          long_t RW                     // register blocking
          >
struct sub_image_pad_1d
{
    static void execute(float const* __restrict i, float* __restrict o,
                        float const* __restrict k, float const* __restrict b,
                        long_t const* __restrict valid_d,
                        long_t const* __restrict valid_h)
    {
        SIMD_FLOAT vout[RW], vwt; // Expected to be in the register file

        ZNN_PRAGMA(unroll(RW))
        for (long_t rw = 0; rw < RW; ++rw)
        {
            vout[rw] =
                conditional_load_or_bias<First>(o + rw * IW::out_stride, b);
        }

        for (long_t kd = valid_d[0]; kd < valid_d[1]; ++kd)
        {
            for (long_t kh = valid_h[0]; kh < valid_h[1]; ++kh)
            {
                for (long_t s = 0; s < IFMs; ++s)
                {
                    for (long_t kw = 0; kw < CW::size; ++kw)
                    {
                        vwt = SIMD_LOAD(
                            k +
                            ((kd * CD::ker_stride + kh * CH::ker_stride +
                              kw * CW::ker_stride) *
                                 SIMD_WIDTH +
                             s) *
                                SIMD_WIDTH);

                        ZNN_PRAGMA(unroll(RW))
                        for (long_t rw = 0; rw < RW; ++rw)
                        {
                            vout[rw] = SIMD_FMADD(
                                vwt,
                                SIMD_SET1(
                                    i[kd * ID::in_stride + kh * IH::in_stride +
                                      (kw + rw * CW::conv_stride) *
                                          IW::in_stride +
                                      s]),
                                vout[rw]);
                        }
                    }
                }
            }
        }

        ZNN_PRAGMA(unroll(RW))
        for (long_t rw = 0; rw < RW; ++rw)
        {
            SIMD_STORE(o + rw * IW::out_stride, vout[rw]);
        }
    }
};

template <bool   First,                 // load or set to bias
          long_t IFMs,                  // number of input images
          class ID, class IH, class IW, // image traits
          class CD, class CH, class CW, // convolution traits
          long_t RW                     // register blocking
          >
struct sub_image_pad
    : std::conditional_t<
          RW == 0, sub_image_pad_dummy,
          sub_image_pad_1d<First, IFMs, ID, IH, IW, CD, CH, CW, RW>>
{
};

} // namespace propagation
} // namespace phi
} // namespace znn
