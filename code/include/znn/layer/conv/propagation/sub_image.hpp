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

/*******************************************************************************
*
* Kernel shape is:
*      k[D][H][W][IN_FMAP][OUT_FMAP] with IN_FMAP=OUT_FMAP=SIMD_WIDTH
*
* input and output shapes are
*      i/o[D][H][W][FMAP] with FMAP=SIMD_WIDTH
*
* if FIRST
*      o[:][:][:][of] is initialized to b[of]
* for of in [0,SIMD_WIDTH), and then
* for all valid d, h, w, kd, kh, kw, if, of
*      o[d][h][w][of] += k[kd][kh][kw][if][of] * i[d+kd][h+kh][w+kw][if];
*
*******************************************************************************/

template <bool   Bias,                    // load or set to bias
          long_t IFMs,                    // number of input images
          class ID, class IH, class IW,   // image traits
          class CD, class CH, class CW,   // convolution traits
          long_t RD, long_t RH, long_t RW // register blocking
          >
struct sub_image;

struct sub_image_dummy
{
    static void execute(float const* __restrict, float* __restrict,
                        float const* __restrict, float const* __restrict)
    {
    }
};

template <bool   Bias,                  // load or set to bias
          long_t IFMs,                  // number of input images
          class ID, class IH, class IW, // image traits
          class CD, class CH, class CW, // convolution traits
          long_t RW                     // register blocking
          >
struct sub_image_1d
{
    static void execute(float const* __restrict i, float* __restrict o,
                        float const* __restrict k, float const* __restrict b)
    {
        SIMD_FLOAT vout[RW], vwt; // Expected to be in the register file

        ZNN_PRAGMA(unroll(RW))
        for (long_t rw = 0; rw < RW; ++rw)
        {
            vout[rw] =
                conditional_load_or_bias<Bias>(o + rw * IW::out_stride, b);
        }

        for (long_t kd = 0; kd < CD::size; ++kd)
        {
            for (long_t kh = 0; kh < CH::size; ++kh)
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

template <bool   Bias,                  // load or set to bias
          long_t IFMs,                  // number of input images
          class ID, class IH, class IW, // image traits
          class CD, class CH, class CW, // convolution traits
          long_t RH, long_t RW          // register blocking
          >
struct sub_image_2d
{
    static void execute(float const* __restrict i, float* __restrict o,
                        float const* __restrict k, float const* __restrict b)
    {
        SIMD_FLOAT vout[RH][RW], vwt; // Expected to be in the register file

        ZNN_PRAGMA(unroll(RH))
        for (long_t rh = 0; rh < RH; ++rh)
        {
            ZNN_PRAGMA(unroll(RW))
            for (long_t rw = 0; rw < RW; ++rw)
            {
                vout[rh][rw] = conditional_load_or_bias<Bias>(
                    o + rh * IH::out_stride + rw * IW::out_stride, b);
            }
        }

        for (long_t kd = 0; kd < CD::size; ++kd)
        {
            for (long_t kh = 0; kh < CH::size; ++kh)
            {
                for (long_t s = 0; s < IFMs; ++s)
                {
                    ZNN_PRAGMA(unroll(CW::size))
                    for (long_t kw = 0; kw < CW::size; ++kw)
                    {
                        vwt = SIMD_LOAD(
                            k +
                            ((kd * CD::ker_stride + kh * CH::ker_stride +
                              kw * CW::ker_stride) *
                                 SIMD_WIDTH +
                             s) *
                                SIMD_WIDTH);

                        ZNN_PRAGMA(unroll_and_jam(RH))
                        for (long_t rh = 0; rh < RH; ++rh)
                        {
                            ZNN_PRAGMA(unroll(RW))
                            for (long_t rw = 0; rw < RW; ++rw)
                            {
                                vout[rh][rw] = SIMD_FMADD(
                                    vwt,
                                    SIMD_SET1(i[kd * ID::in_stride +
                                                (kh + rh * CH::conv_stride) *
                                                    IH::in_stride +
                                                (kw + rw * CW::conv_stride) *
                                                    IW::in_stride +
                                                s]),
                                    vout[rh][rw]);
                            }
                        }
                    }
                }
            }
        }

        ZNN_PRAGMA(unroll(RH))
        for (long_t rh = 0; rh < RH; ++rh)
        {
            ZNN_PRAGMA(unroll(RW))
            for (long_t rw = 0; rw < RW; ++rw)
            {
                SIMD_STORE(o + rh * IH::out_stride + rw * IW::out_stride,
                           vout[rh][rw]);
            }
        }
    }
};

template <bool   Bias,                    // load or set to bias
          long_t IFMs,                    // number of input images
          class ID, class IH, class IW,   // image traits
          class CD, class CH, class CW,   // convolution traits
          long_t RD, long_t RH, long_t RW // register blocking
          >
struct sub_image_3d
{
    static void execute(float const* __restrict i, float* __restrict o,
                        float const* __restrict k, float const* __restrict b)
    {
        SIMD_FLOAT vout[RD][RH][RW], vwt; // Expected to be in the register file

        ZNN_PRAGMA(unroll(RD))
        for (long_t rd = 0; rd < RD; ++rd)
        {
            ZNN_PRAGMA(unroll(RH))
            for (long_t rh = 0; rh < RH; ++rh)
            {
                ZNN_PRAGMA(unroll(RW))
                for (long_t rw = 0; rw < RW; ++rw)
                {
                    vout[rd][rh][rw] = conditional_load_or_bias<Bias>(
                        o + rd * ID::out_stride + rh * IH::out_stride +
                            rw * IW::out_stride,
                        b);
                }
            }
        }

        for (long_t kd = 0; kd < CD::size; ++kd)
        {
            for (long_t kh = 0; kh < CH::size; ++kh)
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

                        ZNN_PRAGMA(unroll(RD))
                        for (long_t rd = 0; rd < RD; ++rd)
                        {
                            ZNN_PRAGMA(unroll(RH))
                            for (long_t rh = 0; rh < RH; ++rh)
                            {
                                ZNN_PRAGMA(unroll(RW))
                                for (long_t rw = 0; rw < RW; ++rw)
                                {
                                    vout[rd][rh][rw] = SIMD_FMADD(
                                        vwt, SIMD_SET1(
                                                 i[(kd + rd * CD::conv_stride) *
                                                       ID::in_stride +
                                                   (kh + rh * CH::conv_stride) *
                                                       IH::in_stride +
                                                   (kw + rw * CW::conv_stride) *
                                                       IW::in_stride +
                                                   s]),
                                        vout[rd][rh][rw]);
                                }
                            }
                        }
                    }
                }
            }
        }

        ZNN_PRAGMA(unroll(RD))
        for (long_t rd = 0; rd < RD; ++rd)
        {
            ZNN_PRAGMA(unroll(RH))
            for (long_t rh = 0; rh < RH; ++rh)
            {
                ZNN_PRAGMA(unroll(RW))
                for (long_t rw = 0; rw < RW; ++rw)
                {
                    SIMD_STORE(o + rd * ID::out_stride + rh * IH::out_stride +
                                   rw * IW::out_stride,
                               vout[rd][rh][rw]);
                }
            }
        }
    }
};

template <bool   Bias,                    // load or set to bias
          long_t IFMs,                    // number of input images
          class ID, class IH, class IW,   // image traits
          class CD, class CH, class CW,   // convolution traits
          long_t RD, long_t RH, long_t RW // register blocking
          >
struct sub_image_3d_vert
{
    static void execute(float const* __restrict i, float* __restrict o,
                        float const* __restrict k, float const* __restrict b)
    {
        SIMD_FLOAT vout[RD][RH][RW], vwt; // Expected to be in the register file

        ZNN_PRAGMA(unroll(RD))
        for (long_t rd = 0; rd < RD; ++rd)
        {
            ZNN_PRAGMA(unroll(RH))
            for (long_t rh = 0; rh < RH; ++rh)
            {
                ZNN_PRAGMA(unroll(RW))
                for (long_t rw = 0; rw < RW; ++rw)
                {
                    vout[rd][rh][rw] = conditional_load_or_bias<Bias>(
                        o + rd * ID::out_stride + rh * IH::out_stride +
                            rw * IW::out_stride,
                        b);
                }
            }
        }
        for (long_t s = 0; s < IFMs; ++s)
        {
            ZNN_PRAGMA(unroll(CD::size))
            for (long_t kd = 0; kd < CD::size; ++kd)
            {
                vwt = SIMD_LOAD(
                    k + ((kd * CD::ker_stride) * SIMD_WIDTH + s) * SIMD_WIDTH);

                ZNN_PRAGMA(unroll(RD))
                for (long_t rd = 0; rd < RD; ++rd)
                {
                    ZNN_PRAGMA(unroll(RH))
                    for (long_t rh = 0; rh < RH; ++rh)
                    {
                        ZNN_PRAGMA(unroll(RW))
                        for (long_t rw = 0; rw < RW; ++rw)
                        {
                            vout[rd][rh][rw] = SIMD_FMADD(
                                vwt, SIMD_SET1(i[(kd + rd * CD::conv_stride) *
                                                     ID::in_stride +
                                                 s]),
                                vout[rd][rh][rw]);
                        }
                    }
                }
            }
        }

        ZNN_PRAGMA(unroll(RD))
        for (long_t rd = 0; rd < RD; ++rd)
        {
            ZNN_PRAGMA(unroll(RH))
            for (long_t rh = 0; rh < RH; ++rh)
            {
                ZNN_PRAGMA(unroll(RW))
                for (long_t rw = 0; rw < RW; ++rw)
                {
                    SIMD_STORE(o + rd * ID::out_stride + rh * IH::out_stride +
                                   rw * IW::out_stride,
                               vout[rd][rh][rw]);
                }
            }
        }
    }
};

template <bool   Bias,                    // load or set to bias
          long_t IFMs,                    // number of input images
          class ID, class IH, class IW,   // image traits
          class CD, class CH, class CW,   // convolution traits
          long_t RD, long_t RH, long_t RW // register blocking
          >
struct sub_image
    : std::conditional_t<
          CW::size == 1 && CH::size == 1,
          sub_image_3d_vert<Bias, IFMs, ID, IH, IW, CD, CH, CW, RD, RH, RW>,
          std::conditional_t<
              RD == 0 || RH == 0 || RW == 0, sub_image_dummy,
              std::conditional_t<
                  RD == 1 && RH == 1,
                  sub_image_1d<Bias, IFMs, ID, IH, IW, CD, CH, CW, RW>,
                  std::conditional_t<
                      RD == 1,
                      sub_image_2d<Bias, IFMs, ID, IH, IW, CD, CH, CW, RH, RW>,
                      sub_image_3d<Bias, IFMs, ID, IH, IW, CD, CH, CW, RD, RH,
                                   RW>>>>>
{
};

} // namespace propagation
} // namespace phi
} // namespace znn
