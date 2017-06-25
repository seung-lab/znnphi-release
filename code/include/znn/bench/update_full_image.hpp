#pragma once

#include "znn/intrin.hpp"
#include "znn/layer/conv/propagation/full_image.hpp"
#include "znn/layer/conv/propagation/full_image_pad.hpp"
#include "znn/layer/conv/propagation/pad_bounds.hpp"
#include "znn/layer/conv/propagation/traits.hpp"
#include "znn/tensor/tensor.hpp"
#include <chrono>
#include <iostream>
#include <string>

namespace znn
{
namespace phi
{

template <bool   First,                    // Load or bias
          long_t OD, long_t OH, long_t OW, // output size
          long_t KD, long_t KH, long_t KW, // kernel size
          long_t SD, long_t SH, long_t SW  // conv stride
          >
inline void benchmark_propagation_full_image(long_t iters = 5)
{
    using namespace propagation;

    static constexpr long_t ID = (OD - 1) * SD + KD;
    static constexpr long_t IH = (OH - 1) * SH + KH;
    static constexpr long_t IW = (OW - 1) * SW + KW;

    host_array<float> in(one_init, ID * IH * IW * SIMD_WIDTH);
    host_array<float> ker(one_init, KD * KH * KW * SIMD_WIDTH * SIMD_WIDTH);
    host_array<float> bias(one_init, SIMD_WIDTH);
    host_array<float> out(one_init, OD * OH * OW * SIMD_WIDTH);

    using work_type =
        full_image<First, SIMD_WIDTH,
                   image_traits<OD, IH * IW * SIMD_WIDTH, OH * OW * SIMD_WIDTH>,
                   image_traits<OH, IW * SIMD_WIDTH, OW * SIMD_WIDTH>,
                   image_traits<OW, SIMD_WIDTH, SIMD_WIDTH>,
                   conv_traits<KD, KW * KH, SD>, conv_traits<KH, KW, SH>,
                   conv_traits<KW, 1, SW>>;

    for (long_t i = 0; i < 2; ++i)
    {
        work_type::execute(in.data(), out.data(), ker.data(), bias.data());
    }

    auto begin = std::chrono::high_resolution_clock::now();

    for (long_t i = 0; i < iters; ++i)
    {
        work_type::execute(in.data(), out.data(), ker.data(), bias.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    double secs   = static_cast<double>(duration) / 1000000;
    double gflops = static_cast<double>(work_type::flops()) / 1000000000;

    std::cout << "bench: " << gflops << "," << (secs / iters) << ","
              << (gflops * iters / secs) << "\n";
}

template <bool   First,                    // Load or bias
          long_t OD, long_t OH, long_t OW, // output size
          long_t KD, long_t KH, long_t KW, // kernel size
          long_t SD, long_t SH, long_t SW, // conv stride
          long_t PD, long_t PH, long_t PW // paddings
          >
inline void benchmark_propagation_full_image_pad(long_t iters = 5)
{
    using namespace propagation;

    static constexpr long_t ID = (OD - 1) * SD + KD;
    static constexpr long_t IH = (OH - 1) * SH + KH;
    static constexpr long_t IW = (OW - 1) * SW + KW;

    host_array<float> in(one_init, ID * IH * IW * SIMD_WIDTH);
    host_array<float> ker(one_init, KD * KH * KW * SIMD_WIDTH * SIMD_WIDTH);
    host_array<float> bias(one_init, SIMD_WIDTH);
    host_array<float> out(one_init, OD * OH * OW * SIMD_WIDTH);

    using work_type = full_image_pad<
        First, SIMD_WIDTH,
        image_traits<OD, IH * IW * SIMD_WIDTH, OH * OW * SIMD_WIDTH>,
        image_traits<OH, IW * SIMD_WIDTH, OW * SIMD_WIDTH>,
        image_traits<OW, SIMD_WIDTH, SIMD_WIDTH>, conv_traits<KD, KW * KH, SD>,
        conv_traits<KH, KW, SH>, conv_traits<KW, 1, SW>>;

    auto valid_d = pad_bounds<OD, KD, PD>::bounds;
    auto valid_h = pad_bounds<OH, KH, PH>::bounds;

    for (long_t i = 0; i < 2; ++i)
    {
        work_type::execute(in.data(), out.data(), ker.data(), bias.data(),
                           valid_d, valid_h);
    }

    auto begin = std::chrono::high_resolution_clock::now();

    for (long_t i = 0; i < iters; ++i)
    {
        work_type::execute(in.data(), out.data(), ker.data(), bias.data(),
                           valid_d, valid_h);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    double secs   = static_cast<double>(duration) / 1000000;
    double gflops = static_cast<double>(work_type::flops()) / 1000000000;

    std::cout << "bench: " << gflops << "," << (secs / iters) << ","
              << (gflops * iters / secs) << "\n";
}

} // namespace phi
} // namespace znn
