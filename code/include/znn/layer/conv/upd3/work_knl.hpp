#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/conv/upd3/blocking.hpp"
#include "znn/util/conditional_load.hpp"

// kernel shape is
// K[D][H][W][IN_FMAP][OUT_FMAP] with IN=OUT=SIMD_WIDTH

// input and output shapes are
// Im[D][H][W][FMAP]

// for all valid d, h, w, kd, kh, kw, if, of
// Out[d][h][w][of] += K[kd][kh][kw][if][of] * In[d+kd][h+kh][w+kw][if];

// K[if][kd][kh][kw][of] += In[d][h][w][of] * Out[d+kd][h+kh][w+kw][if];

namespace znn
{
namespace phi
{

template <class D, class H, class W,   // out size and in/out strides
          class CD, class CH, class CW // convolution params
          >
class upd_work
{
private:
    using RB = upd_blocking_t<CD::s, CH::s, CW::s>;

    static const long_t SW = SIMD_WIDTH;

    // might not need this!
    // static void set_all_zero( float * __restrict k )
    // {
    //     SIMD_FMADD zero = SIMD_ZERO();
    //     for ( long_t i = 0; i < CD::s*CH::s*CW::s*SIMD_WIDTH )
    //     {
    //         SIMD_STORE( k + i * SIMD_WIDTH, zero );
    //     }
    // }

public:
    template <long_t RBF, long_t RBD, long_t RBH, long_t RBW, long_t MAXW>
    static typename std::enable_if<
        RBF == 0 || RBD == 0 || RBH == 0 || RBW == 0 || MAXW == 0, void>::type
    chunk(float const* __restrict, float const* __restrict, float* __restrict)
    {
    }

    template <long_t RBF, long_t RBD, long_t RBH, long_t RBW, long_t MAXW>
    static typename std::enable_if<(RBF > 0) && (RBD > 0) && (RBH > 0) &&
                                       (RBW > 0) && (MAXW > 0) && (MAXW % 2),
                                   void>::type
    chunk(float const* __restrict i, float const* __restrict o,
          float* __restrict k)
    {
        SIMD_FLOAT kout[RBD][RBH][RBW][RBF], kwt; // in registers

#pragma unroll(RBD)
        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                    for (long_t rbf = 0; rbf < RBF; ++rbf)
                    {
                        kout[rbd][rbh][rbw][rbf] = conditional_load<false>(
                            k +
                            (rbf * CD::s * CH::s * CW::s + rbd * CH::s * CW::s +
                             rbh * CW::s + rbw) *
                                SIMD_WIDTH);
                    }

#pragma unroll
        for (long_t w = 0; w < MAXW; ++w)

        {
            kwt = SIMD_LOAD(o + (w * W::os) * SIMD_WIDTH);
#pragma unroll(RBD)
            for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                    for (long_t rbw = 0; rbw < RBW; ++rbw)
                    {
                        SIMD_PREFETCH_L1(reinterpret_cast<const char*>(
                            i +
                            ((w + rbw) * W::is + rbd * D::is + rbh * H::is) *
                                SIMD_WIDTH));
#pragma unroll(RBF)
                        for (long_t rbf = 0; rbf < RBF; ++rbf)
                        {
                            kout[rbd][rbh][rbw][rbf] = SIMD_FMADD(
                                kwt, SIMD_SET1(i[((w + rbw) * W::is +
                                                  rbd * D::is + rbh * H::is) *
                                                     SIMD_WIDTH +
                                                 rbf]),
                                kout[rbd][rbh][rbw][rbf]);
                        }
                    }
        }

#pragma unroll(RBD)
        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                    for (long_t rbf = 0; rbf < RBF; ++rbf)
                    {
                        SIMD_STORE(k +
                                       (rbf * CD::s * CH::s * CW::s +
                                        rbd * CH::s * CW::s + rbh * CW::s +
                                        rbw) *
                                           SIMD_WIDTH,
                                   kout[rbd][rbh][rbw][rbf]);
                    }
    }

    template <long_t RBF, long_t RBD, long_t RBH, long_t RBW, long_t MAXW>
    static typename std::enable_if<(RBF > 0) && (RBD > 0) && (RBH > 0) &&
                                       (RBW > 0) && (MAXW > 0) &&
                                       (MAXW % 2 == 0) && (MAXW % 4),
                                   void>::type
    chunk(float const* __restrict i, float const* __restrict o,
          float* __restrict k)
    {
        SIMD_FLOAT kout[RBD][RBH][RBW][RBF], kwt; // in registers

#pragma unroll(RBD)
        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                    for (long_t rbf = 0; rbf < RBF; ++rbf)
                    {
                        kout[rbd][rbh][rbw][rbf] = conditional_load<false>(
                            k +
                            (rbf * CD::s * CH::s * CW::s + rbd * CH::s * CW::s +
                             rbh * CW::s + rbw) *
                                SIMD_WIDTH);
                    }

        for (long_t w = 0; w < MAXW / 2; ++w)

        {
            kwt = SIMD_LOAD(o + (w * W::os * 2) * SIMD_WIDTH);

#pragma unroll(RBD)
            for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                    for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                        for (long_t rbf = 0; rbf < RBF; ++rbf)
                        {
                            kout[rbd][rbh][rbw][rbf] = SIMD_FMADD(
                                kwt, SIMD_SET1(i[((w * 2 + rbw) * W::is +
                                                  rbd * D::is + rbh * H::is) *
                                                     SIMD_WIDTH +
                                                 rbf]),
                                kout[rbd][rbh][rbw][rbf]);
                        }

            kwt = SIMD_LOAD(o + ((w * 2 + 1) * W::os) * SIMD_WIDTH);

#pragma unroll(RBD)
            for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                    for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                        for (long_t rbf = 0; rbf < RBF; ++rbf)
                        {
                            kout[rbd][rbh][rbw][rbf] = SIMD_FMADD(
                                kwt, SIMD_SET1(i[((w * 2 + 1 + rbw) * W::is +
                                                  rbd * D::is + rbh * H::is) *
                                                     SIMD_WIDTH +
                                                 rbf]),
                                kout[rbd][rbh][rbw][rbf]);
                        }
        }

#pragma unroll(RBD)
        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                    for (long_t rbf = 0; rbf < RBF; ++rbf)
                    {
                        SIMD_STORE(k +
                                       (rbf * CD::s * CH::s * CW::s +
                                        rbd * CH::s * CW::s + rbh * CW::s +
                                        rbw) *
                                           SIMD_WIDTH,
                                   kout[rbd][rbh][rbw][rbf]);
                    }
    }

    template <long_t RBF, long_t RBD, long_t RBH, long_t RBW, long_t MAXW>
    static
        typename std::enable_if<(RBF > 0) && (RBD > 0) && (RBH > 0) &&
                                    (RBW > 0) && (MAXW > 0) && (MAXW % 4 == 0),
                                void>::type
        chunk(float const* __restrict i, float const* __restrict o,
              float* __restrict k)
    {
        SIMD_FLOAT kout[RBD][RBH][RBW][RBF], kwt; // in registers

#pragma unroll(RBD)
        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                    for (long_t rbf = 0; rbf < RBF; ++rbf)
                    {
                        kout[rbd][rbh][rbw][rbf] = conditional_load<false>(
                            k +
                            (rbf * CD::s * CH::s * CW::s + rbd * CH::s * CW::s +
                             rbh * CW::s + rbw) *
                                SIMD_WIDTH);
                    }

        for (long_t w = 0; w < MAXW / 4; ++w)

        {
            kwt = SIMD_LOAD(o + (w * W::os * 4) * SIMD_WIDTH);

#pragma unroll(RBD)
            for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                    for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                        for (long_t rbf = 0; rbf < RBF; ++rbf)
                        {
                            kout[rbd][rbh][rbw][rbf] = SIMD_FMADD(
                                kwt, SIMD_SET1(i[((w * 4 + rbw) * W::is +
                                                  rbd * D::is + rbh * H::is) *
                                                     SIMD_WIDTH +
                                                 rbf]),
                                kout[rbd][rbh][rbw][rbf]);
                        }

            kwt = SIMD_LOAD(o + ((w * 4 + 1) * W::os) * SIMD_WIDTH);

#pragma unroll(RBD)
            for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                    for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                        for (long_t rbf = 0; rbf < RBF; ++rbf)
                        {
                            kout[rbd][rbh][rbw][rbf] = SIMD_FMADD(
                                kwt, SIMD_SET1(i[((w * 4 + 1 + rbw) * W::is +
                                                  rbd * D::is + rbh * H::is) *
                                                     SIMD_WIDTH +
                                                 rbf]),
                                kout[rbd][rbh][rbw][rbf]);
                        }

            kwt = SIMD_LOAD(o + ((w * 4 + 2) * W::os) * SIMD_WIDTH);

#pragma unroll(RBD)
            for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                    for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                        for (long_t rbf = 0; rbf < RBF; ++rbf)
                        {
                            kout[rbd][rbh][rbw][rbf] = SIMD_FMADD(
                                kwt, SIMD_SET1(i[((w * 4 + 2 + rbw) * W::is +
                                                  rbd * D::is + rbh * H::is) *
                                                     SIMD_WIDTH +
                                                 rbf]),
                                kout[rbd][rbh][rbw][rbf]);
                        }

            kwt = SIMD_LOAD(o + ((w * 4 + 3) * W::os) * SIMD_WIDTH);

#pragma unroll(RBD)
            for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
                for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                    for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                        for (long_t rbf = 0; rbf < RBF; ++rbf)
                        {
                            kout[rbd][rbh][rbw][rbf] = SIMD_FMADD(
                                kwt, SIMD_SET1(i[((w * 4 + 3 + rbw) * W::is +
                                                  rbd * D::is + rbh * H::is) *
                                                     SIMD_WIDTH +
                                                 rbf]),
                                kout[rbd][rbh][rbw][rbf]);
                        }
        }

#pragma unroll(RBD)
        for (long_t rbd = 0; rbd < RBD; ++rbd)
#pragma unroll(RBH)
            for (long_t rbh = 0; rbh < RBH; ++rbh)
#pragma unroll(RBW)
                for (long_t rbw = 0; rbw < RBW; ++rbw)
#pragma unroll(RBF)
                    for (long_t rbf = 0; rbf < RBF; ++rbf)
                    {
                        SIMD_STORE(k +
                                       (rbf * CD::s * CH::s * CW::s +
                                        rbd * CH::s * CW::s + rbh * CW::s +
                                        rbw) *
                                           SIMD_WIDTH,
                                   kout[rbd][rbh][rbw][rbf]);
                    }
    }

    static long_t flops()
    {
        return CW::s * CH::s * CD::s * SW * D::s * H::s * W::s * SIMD_WIDTH * 2;
    }

    template <long_t RBD, long_t RBH, long_t RBW, long_t MW>
    static void loopf(float const* __restrict i, float const* __restrict o,
                      float* __restrict k)
    {
        static const long_t FULL = SW / RB::f;
        static const long_t PART = SW % RB::f;

        for (long_t d = 0; d < FULL; ++d)
        {
            upd_work::template chunk<RB::f, RBD, RBH, RBW, MW>(i, o, k);
            i += RB::f;
            k += RB::f * CD::s * CH::s * CW::s * SIMD_WIDTH;
        }

        if (PART)
        {
            upd_work::template chunk<RB::f, RBD, RBH, RBW, MW>(i, o, k);
        }
    }

    template <long_t RBD, long_t RBH, long_t MW>
    static void loopw(float const* __restrict i, float const* __restrict o,
                      float* __restrict k)
    {
        static const long_t FULL = CW::s / RB::w;
        static const long_t PART = CW::s % RB::w;

        for (long_t d = 0; d < FULL; ++d)
        {
            upd_work::template loopf<RBD, RBH, RB::w, MW>(i, o, k);
            i += RB::w * W::is * SIMD_WIDTH;
            k += RB::w * SIMD_WIDTH;
        }

        if (PART)
        {
            upd_work::template loopf<RBD, RBH, PART, MW>(i, o, k);
        }
    }

    template <long_t RBD, long_t MW>
    static void looph(float const* __restrict i, float const* __restrict o,
                      float* __restrict k)
    {
        static const long_t FULL = CH::s / RB::h;
        static const long_t PART = CH::s % RB::h;

        for (long_t d = 0; d < FULL; ++d)
        {
            upd_work::template loopw<RBD, RB::h, MW>(i, o, k);
            i += RB::h * H::is * SIMD_WIDTH;
            k += RB::h * CW::s * SIMD_WIDTH;
        }

        if (PART)
        {
            upd_work::template loopw<RBD, PART, MW>(i, o, k);
        }
    }

    template <long_t MW>
    static void      loopd(float const* __restrict i, float const* __restrict o,
                      float* __restrict k)
    {
        static const long_t FULL = CD::s / RB::d;
        static const long_t PART = CD::s % RB::d;

        for (long_t d = 0; d < FULL; ++d)
        {
            upd_work::template looph<RB::d, MW>(i, o, k);
            i += RB::d * D::is * SIMD_WIDTH;
            k += RB::d * CH::s * CW::s * SIMD_WIDTH;
        }

        if (PART)
        {
            upd_work::template looph<PART, MW>(i, o, k);
        }
    }

    static void loopw(float const* __restrict i, float const* __restrict o,
                      float* __restrict k)
    {
        static const long_t FULL = W::s / 128;
        static const long_t PART = W::s % 128;

        for (long_t w = 0; w < FULL; ++w)
        {
            for (long_t h = 0; h < H::s; ++h)
            {
                upd_work::template loopd<128>(i + h * H::is * SIMD_WIDTH,
                                              o + h * H::os * SIMD_WIDTH, k);
            }
            i += 128 * W::is * SIMD_WIDTH;
            o += 128 * W::os * SIMD_WIDTH;
        }

        if (PART)
        {
            for (long_t h = 0; h < H::s; ++h)
            {
                upd_work::template loopd<PART>(i + h * H::is * SIMD_WIDTH,
                                               o + h * H::os * SIMD_WIDTH, k);
            }
        }
    }

    static void execute(float const* __restrict i, float const* __restrict o,
                        float* __restrict k)
    {

        for (long_t d = 0; d < D::s; ++d)
        {
            loopw(i + (d * D::is) * SIMD_WIDTH, o + (d * D::os) * SIMD_WIDTH,
                  k);
        }
    }
};
}
} // namespace znn:phi
