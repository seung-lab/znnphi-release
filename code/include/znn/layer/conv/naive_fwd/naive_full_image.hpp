#pragma once

#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/types.hpp"

namespace znn
{
namespace phi
{

/*******************************************************************************
/*
/* Kernel shape is:
/*      k[D][H][W][IN_FMAP][OUT_FMAP] with IN_FMAP=OUT_FMAP=SIMD_WIDTH
/*
/* input and output shapes are
/*      i/o[D][H][W][FMAP] with FMAP=SIMD_WIDTH
/*
/* if FIRST
/*      o[:][:][:][of] is initialized to b[of]
/* for of in [0,SIMD_WIDTH), and then
/* for all valid d, h, w, kd, kh, kw, if, of
/*      o[d][h][w][of] += k[kd][kh][kw][if][of] * i[d+kd][h+kh][w+kw][if];
/*
*******************************************************************************/

template <bool   FIRST,                // load or set to the bias
          long_t IFMS,                 // number of input featuremaps
          class D, class H, class W,   // out size and in/out strides
          class CD, class CH, class CW // convolution params
          >
class naive_full_image
{
public:
    static void execute(float const* i, float* o, float const* k,
                        float const* b)
    {
        for (long_t ofm = 0; ofm < SIMD_WIDTH; ++ofm)
        {
            for (long_t ofd = 0; ofd < D::s; ++ofd)
            {
                for (long_t ofh = 0; ofh < H::s; ++ofh)
                {
                    for (long_t ofw = 0; ofw < W::s; ++ofw)
                    {
                        if (FIRST)
                        {
                            o[ofd * D::os + ofh * H::os + ofw * W::os + ofm] =
                                b[ofm];
                        }
                        for (long_t ifm = 0; ifm < IFMS; ++ifm)
                        {
                            for (long_t kd = 0; kd < CD::s; ++kd)
                            {
                                for (long_t kh = 0; kh < CH::s; ++kh)
                                {
                                    for (long_t kw = 0; kw < CW::s; ++kw)
                                    {
                                        o[ofd * D::os + ofh * H::os +
                                          ofw * W::os + ofm] +=
                                            i[(ofd + kd) * D::is +
                                              (ofh + kh) * H::is +
                                              (ofw + kw) * W::is + ifm] *
                                            k[((kd * CW::s * CH::s +
                                                kh * CW::s + kw) *
                                                   SIMD_WIDTH +
                                               ifm) *
                                                  SIMD_WIDTH +
                                              ofm];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};
} // namespace phi
} // namespace znn
