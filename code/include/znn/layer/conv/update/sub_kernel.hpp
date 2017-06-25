#pragma once

#include "znn/intrin.hpp"
#include "znn/types.hpp"

#include <type_traits>

namespace znn
{
namespace phi
{
namespace update
{

/*******************************************************************************
*
*
*******************************************************************************/

template <class ID, class IH, class IW,               // image traits
          class CD, class CH, class CW,               // convolution traits
          long_t MAXW,                                // length along W
          long_t RS, long_t RD, long_t RH, long_t RW> // register blocking
struct sub_kernel;

struct sub_kernel_dummy
{
    static void execute(float const* __restrict, float const* __restrict,
                        float* __restrict)
    {
    }
};

template <class ID, class IH, class IW,               // image traits
          class CD, class CH, class CW,               // convolution traits
          long_t MAXW,                                // length along W
          long_t RS, long_t RD, long_t RH, long_t RW> // register blocking
struct sub_kernel_full
{
    static void execute(float const* __restrict i, float const* __restrict o,
                        float* __restrict k)
    {
        SIMD_FLOAT kout[RD][RH][RW][RS], kwt; // In the register file

        for (long_t ds = 0; ds < SIMD_WIDTH / RS; ++ds)
        {

            ZNN_PRAGMA(unroll(RD))
            for (long_t rd = 0; rd < RD; ++rd)
            {
                ZNN_PRAGMA(unroll(RH))
                for (long_t rh = 0; rh < RH; ++rh)
                {
                    ZNN_PRAGMA(unroll(RW))
                    for (long_t rw = 0; rw < RW; ++rw)
                    {
                        ZNN_PRAGMA(unroll(RS))
                        for (long_t rs = 0; rs < RS; ++rs)
                        {
                            kout[rd][rh][rw][rs] = SIMD_LOAD(
                                k +
                                ((rw * CW::ker_stride + rh * CH::ker_stride +
                                  rd * CD::ker_stride) *
                                     SIMD_WIDTH +
                                 rs) *
                                    SIMD_WIDTH);
                        }
                    }
                }
            }

            // We perform some manual partial unrolling
            for (long_t w = 0; w < MAXW / 4; ++w)
            {
                kwt = SIMD_LOAD(o + (w * 4) * IW::out_stride);

                ZNN_PRAGMA(unroll(RD))
                for (long_t rd = 0; rd < RD; ++rd)
                {
                    ZNN_PRAGMA(unroll(RH))
                    for (long_t rh = 0; rh < RH; ++rh)
                    {
                        ZNN_PRAGMA(unroll(RS))
                        for (long_t rs = 0; rs < RS; ++rs)
                        {
                            ZNN_PRAGMA(unroll(RW))
                            for (long_t rw = 0; rw < RW; ++rw)
                            {
                                kout[rd][rh][rw][rs] = SIMD_FMADD(
                                    kwt,
                                    SIMD_SET1(i[(w * 4 + rw) * IW::in_stride +
                                                rh * IH::in_stride +
                                                rd * ID::in_stride + rs]),
                                    kout[rd][rh][rw][rs]);
                            }
                        }
                    }
                }

                kwt = SIMD_LOAD(o + (w * 4 + 1) * IW::out_stride);

                ZNN_PRAGMA(unroll(RD))
                for (long_t rd = 0; rd < RD; ++rd)
                {
                    ZNN_PRAGMA(unroll(RH))
                    for (long_t rh = 0; rh < RH; ++rh)
                    {
                        ZNN_PRAGMA(unroll(RS))
                        for (long_t rs = 0; rs < RS; ++rs)
                        {
                            ZNN_PRAGMA(unroll(RW))
                            for (long_t rw = 0; rw < RW; ++rw)
                            {
                                kout[rd][rh][rw][rs] = SIMD_FMADD(
                                    kwt,
                                    SIMD_SET1(
                                        i[(w * 4 + 1 + rw) * IW::in_stride +
                                          rh * IH::in_stride +
                                          rd * ID::in_stride + rs]),
                                    kout[rd][rh][rw][rs]);
                            }
                        }
                    }
                }

                kwt = SIMD_LOAD(o + (w * 4 + 2) * IW::out_stride);

                ZNN_PRAGMA(unroll(RD))
                for (long_t rd = 0; rd < RD; ++rd)
                {
                    ZNN_PRAGMA(unroll(RH))
                    for (long_t rh = 0; rh < RH; ++rh)
                    {
                        ZNN_PRAGMA(unroll(RS))
                        for (long_t rs = 0; rs < RS; ++rs)
                        {
                            ZNN_PRAGMA(unroll(RW))
                            for (long_t rw = 0; rw < RW; ++rw)
                            {
                                kout[rd][rh][rw][rs] = SIMD_FMADD(
                                    kwt,
                                    SIMD_SET1(
                                        i[(w * 4 + 2 + rw) * IW::in_stride +
                                          rh * IH::in_stride +
                                          rd * ID::in_stride + rs]),
                                    kout[rd][rh][rw][rs]);
                            }
                        }
                    }
                }

                kwt = SIMD_LOAD(o + (w * 4 + 3) * IW::out_stride);

                ZNN_PRAGMA(unroll(RD))
                for (long_t rd = 0; rd < RD; ++rd)
                {
                    ZNN_PRAGMA(unroll(RH))
                    for (long_t rh = 0; rh < RH; ++rh)
                    {
                        ZNN_PRAGMA(unroll(RS))
                        for (long_t rs = 0; rs < RS; ++rs)
                        {
                            ZNN_PRAGMA(unroll(RW))
                            for (long_t rw = 0; rw < RW; ++rw)
                            {
                                kout[rd][rh][rw][rs] = SIMD_FMADD(
                                    kwt,
                                    SIMD_SET1(
                                        i[(w * 4 + 3 + rw) * IW::in_stride +
                                          rh * IH::in_stride +
                                          rd * ID::in_stride + rs]),
                                    kout[rd][rh][rw][rs]);
                            }
                        }
                    }
                }
            }

            if (MAXW % 4)
            {
                float const* __restrict o2 =
                    o + (MAXW - (MAXW % 4)) * IW::out_stride;
                float const* __restrict i2 =
                    i + (MAXW - (MAXW % 4)) * IW::in_stride;

                for (long_t w2 = 0; w2 < (MAXW % 4); ++w2)
                {
                    kwt = SIMD_LOAD(o2 + w2 * IW::out_stride);

                    ZNN_PRAGMA(unroll(RD))
                    for (long_t rd = 0; rd < RD; ++rd)
                    {
                        ZNN_PRAGMA(unroll(RH))
                        for (long_t rh = 0; rh < RH; ++rh)
                        {
                            ZNN_PRAGMA(unroll(RS))
                            for (long_t rs = 0; rs < RS; ++rs)
                            {
                                ZNN_PRAGMA(unroll(RW))
                                for (long_t rw = 0; rw < RW; ++rw)
                                {
                                    kout[rd][rh][rw][rs] = SIMD_FMADD(
                                        kwt,
                                        SIMD_SET1(i2[(w2 + rw) * IW::in_stride +
                                                     rh * IH::in_stride +
                                                     rd * ID::in_stride + rs]),
                                        kout[rd][rh][rw][rs]);
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
                        ZNN_PRAGMA(unroll(RS))
                        for (long_t rs = 0; rs < RS; ++rs)
                        {
                            SIMD_STORE(k +
                                           ((rw * CW::ker_stride +
                                             rh * CH::ker_stride +
                                             rd * CD::ker_stride) *
                                                SIMD_WIDTH +
                                            rs) *
                                               SIMD_WIDTH,
                                       kout[rd][rh][rw][rs]);
                        }
                    }
                }
            }

            i += RS;
            k += RS * SIMD_WIDTH;
        }
    }
};

template <class ID, class IH, class IW,               // image traits
          class CD, class CH, class CW,               // convolution traits
          long_t MAXW,                                // length along W
          long_t RS, long_t RD, long_t RH, long_t RW> // register blocking
struct sub_kernel
    : std::conditional_t<
          RS == 0 || RD == 0 || RH == 0 || RW == 0, sub_kernel_dummy,
          sub_kernel_full<ID, IH, IW, CD, CH, CW, MAXW, RS, RD, RH, RW>>
{
};

} // namespace update
} // namespace phi
} // namespace znn
