#pragma once

#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/conv/fwd4/blocking.hpp"
#include "znn/layer/task.hpp"
#include "znn/types.hpp"
#include "znn/util/conditional_load.hpp"

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
class fwd_work
{
private:
    using RB = fwd_blocking_t<D::s, H::s, W::s>;

public:
    // Special case when the number of blocked registers is zero,
    // we need this to simplify the implementation, stressing the compiler
    // as little as possible
    template <long_t RBD, long_t RBH, long_t RBW>
    static typename std::enable_if<RBD == 0 || RBH == 0 || RBW == 0, void>::type
    chunk(float const* __restrict, float* __restrict, float const* __restrict,
          float const* __restrict)
    {
    }

    // Special case when the blocking is only along the least significant
    // dimension.  Turns out that having dummy loops (0 to 0) prevents both
    // (older versions) GCC and ICC to optimally compile the code for large
    // values of RBD*RBH*RBW. There might be a way to simpift the implementation
    // by using some compiler switches. Not necessary for newer compilers
    template <long_t RBD, long_t RBH, long_t RBW>
    static typename std::enable_if<(RBD == 1) && (RBH == 1) && (RBW > 0),
                                   void>::type
    chunk(float const* __restrict i, float* __restrict o,
          float const* __restrict k, float const* __restrict b)

    {
        SIMD_FLOAT vout[RBW], vwt; // Expected to be in the register file

#pragma unroll(RBW)
        for (long_t rbw = 0; rbw < RBW; ++rbw)
        {
            vout[rbw] = conditional_load_or_bias<FIRST>(o + rbw * W::os, b);
        }

        for (long_t kd = 0; kd < CD::s; ++kd)

            for (long_t kh = 0; kh < CH::s; ++kh)

                for (long_t s = 0; s < IFMS; ++s)

                    for (long_t kw = 0; kw < CW::s; ++kw)
                    {
                        vwt =
                            SIMD_LOAD(k +
                                      ((kh * CW::s + kw + kd * CW::s * CH::s) *
                                           SIMD_WIDTH +
                                       s) *
                                          SIMD_WIDTH);

#pragma unroll(RBW)
                        for (long_t rbw = 0; rbw < RBW; ++rbw)
                        {
                            vout[rbw] = SIMD_FMADD(
                                vwt, SIMD_SET1(i[(kd * D::is + kh * H::is +
                                                  (kw + rbw) * W::is) +
                                                 s]),
                                vout[rbw]);
                        }
                    }

#pragma unroll(RBW)
        for (long_t rbw = 0; rbw < RBW; ++rbw)
        {
            SIMD_STORE(o + rbw * W::os, vout[rbw]);
        }
    }

    // Generic case
    template <long_t RBD, long_t RBH, long_t RBW>
    static typename std::enable_if<((RBD > 1) || (RBH > 1)) && (RBD > 0) &&
                                       (RBH > 0) && (RBW > 0),
                                   void>::type
    chunk(float const* __restrict i, float* __restrict o,
          float const* __restrict k, float const* __restrict b)

    {
        SIMD_FLOAT vout[RBD][RBH][RBW], vwt; // in registers

#pragma unroll(RBD)
        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                for (long_t rbw = 0; rbw < RBW; ++rbw)
                {
                    vout[rbd][rbh][rbw] = conditional_load_or_bias<FIRST>(
                        o + (rbw * W::os + rbd * D::os + rbh * H::os), b);
                }

        for (long_t kd = 0; kd < CD::s; ++kd)

            for (long_t kh = 0; kh < CH::s; ++kh)

                for (long_t s = 0; s < IFMS; ++s)

                    for (long_t kw = 0; kw < CW::s; ++kw)
                    {
                        vwt =
                            SIMD_LOAD(k +
                                      ((kh * CW::s + kw + kd * CW::s * CH::s) *
                                           SIMD_WIDTH +
                                       s) *
                                          SIMD_WIDTH);

#pragma unroll(RBD)
                        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                                for (long_t rbw = 0; rbw < RBW; ++rbw)
                                {
                                    vout[rbd][rbh][rbw] = SIMD_FMADD(
                                        vwt, SIMD_SET1(i[((kd * CD::dilation +
                                                           rbd * CD::stride) *
                                                              D::is +
                                                          (kh * CH::dilation +
                                                           rbh * CH::stride) *
                                                              H::is +
                                                          (kw * CW::dilation +
                                                           rbw * CW::stride) *
                                                              W::is) +
                                                         s]),
                                        vout[rbd][rbh][rbw]);
                                }
                    }

#pragma unroll(RBD)
        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                for (long_t rbw = 0; rbw < RBW; ++rbw)
                {
                    SIMD_STORE(o + (rbw * W::os + rbd * D::os + rbh * H::os),
                               vout[rbd][rbh][rbw]);
                }
    }

    static long_t flops()
    {
        return CW::s * CH::s * CD::s * IFMS * D::s * H::s * W::s * SIMD_WIDTH *
               2;
    }

    template <long_t DD, long_t HH>
    static void loopw(float const* __restrict i, float* __restrict o,
                      float const* __restrict k, float const* __restrict b)
    {
        static const long_t FULL = W::s / RB::w;
        static const long_t PART = W::s % RB::w;

        for (long_t d = 0; d < FULL; ++d)
        {
            fwd_work::template chunk<DD, HH, RB::w>(i, o, k, b);
            i += RB::w * W::is * CW::stride;
            o += RB::w * W::os;
        }

        if (PART)
        {
            fwd_work::template chunk<DD, HH, PART>(i, o, k, b);
        }
    }

    template <long_t DD>
    static void      looph(float const* __restrict i, float* __restrict o,
                      float const* __restrict k, float const* __restrict b)
    {
        static const long_t FULL = H::s / RB::h;
        static const long_t PART = H::s % RB::h;

        for (long_t d = 0; d < FULL; ++d)
        {
            fwd_work::template loopw<DD, RB::h>(i, o, k, b);
            i += RB::h * H::is * CH::stride;
            o += RB::h * H::os;
        }

        if (PART)
        {
            fwd_work::template loopw<DD, PART>(i, o, k, b);
        }
    }

    static void loopd(float const* __restrict i, float* __restrict o,
                      float const* __restrict k, float const* __restrict b)
    {
        static const long_t FULL = D::s / RB::d;
        static const long_t PART = D::s % RB::d;

        for (long_t d = 0; d < FULL; ++d)
        {
            fwd_work::template looph<RB::d>(i, o, k, b);
            i += RB::d * D::is * CD::stride;
            o += RB::d * D::os;
        }

        if (PART)
        {
            fwd_work::template looph<PART>(i, o, k, b);
        }
    }

public:
    static execute(float const* i, float* o, float const* k, float const* b)
    {
        loopd(i, o, k, b);
    }
};

// Help the compiler with a dummy special case.  Not necessary for correctness.
template <bool   FIRST,                // load or set to zero
          long_t IFMS,                 // number of input featuremaps
          class D, class H, class W,   // out size and in/out strides
          class CD, class CH, class CW // convolution params
          >
class fwd_work<FIRST, 0, D, H, W, CD, CH, CW>
{

public:
    static long_t flops() { return 0; }

    static execute(float const*, float*, float const*, float const*) {}
};
}
} // namespace znn:phi
