#include "znn/layer/conv/fwd4/naive_work.hpp"
#include "znn/layer/conv/fwd4/work.hpp"
#include "znn/layer/dimension.hpp"
#include "znn/tensor/tensor.hpp"

#include <cmath>
#include <iostream>

using namespace znn::phi;

inline float max_abs_diff(float const* a, float const* b, long_t N)
{
    float r = 0;
    for (long_t i = 0; i < N; ++i)
    {
        r = std::max(r, std::abs(a[i] - b[i]));
    }
    return r;
}

template <bool First, long_t ID, long_t IH, long_t IW, long_t KD, long_t KH,
          long_t KW>
inline void      test_single_forward()
{
    using CD = conv_traits<KD, 1, 1>;
    using CH = conv_traits<KH, 1, 1>;
    using CW = conv_traits<KW, 1, 1>;

    static constexpr long_t OD = ID - KD + 1;
    static constexpr long_t OH = IH - KH + 1;
    static constexpr long_t OW = IW - KW + 1;

    using DT = dimension<OD, IH * IW * SIMD_WIDTH, OH * OW * SIMD_WIDTH>;
    using HT = dimension<OH, IW * SIMD_WIDTH, OW * SIMD_WIDTH>;
    using WT = dimension<OW, SIMD_WIDTH, SIMD_WIDTH>;

    using fast_t = fwd_work<First, SIMD_WIDTH, DT, HT, WT, CD, CH, CW>;
    using slow_t = naive_fwd_work<First, SIMD_WIDTH, DT, HT, WT, CD, CH, CW>;

    host_array<float> in(rand_init, ID * IH * IW * SIMD_WIDTH);
    host_array<float> ker(rand_init, KD * KH * KW * SIMD_WIDTH * SIMD_WIDTH);
    host_array<float> b(rand_init, SIMD_WIDTH);
    host_array<float> out1(rand_init, OD * OH * OW * SIMD_WIDTH);
    host_array<float> out2(one_init, OD * OH * OW * SIMD_WIDTH);

    out2 = out1;

    fast_t fast;
    slow_t::execute(in.data(), out2.data(), ker.data(), b.data());
    fast.execute(in.data(), out1.data(), ker.data(), b.data());

    std::cout << max_abs_diff(out1.data(), out2.data(),
                              OD * OH * OW * SIMD_WIDTH)
              << "\n";
}

int main()
{
    //test_single_forward<true,1,5,5,1,3,3>();
    test_single_forward<false,1,100,100,1,3,3>();
    test_single_forward<true,1,100,100,1,3,3>();
    test_single_forward<false,1,4,4,1,4,4>();
}
