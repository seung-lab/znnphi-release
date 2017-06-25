#include "znn/intrin.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/layer/dimension.hpp"
#include <chrono>
#include <iostream>
#include <string>

namespace znn
{
namespace phi
{

template <long_t OD, long_t OH, long_t OW, // output size
          long_t KD, long_t KH, long_t KW> // kernel size
inline void
benchmark_update_full_kernel(long_t iters = 5)
{
    static constexpr long_t ID = (OD - 1) * 1 + KD;
    static constexpr long_t IH = (OH - 1) * 1 + KH;
    static constexpr long_t IW = (OW - 1) * 1 + KW;

    host_array<float> in(one_init, ID * IH * IW * SIMD_WIDTH);
    host_array<float> ker(one_init, KD * KH * KW * SIMD_WIDTH * SIMD_WIDTH);
    host_array<float> out(one_init, OD * OH * OW * SIMD_WIDTH);

    using work_type = upd_work<
        dimension<OD, IH * IW , OH * OW>,
        dimension<OH, IW, OW>,
        dimension<OW, 1, 1>, conv_traits<KD, KW * KH, 1>,
        conv_traits<KH, KW, 1>, conv_traits<KW, 1, 1>>;

    for (long_t i = 0; i < iters; ++i)
    {
        work_type::execute(in.data(), out.data(), ker.data());
    }

    auto begin = std::chrono::high_resolution_clock::now();

    for (long_t i = 0; i < iters; ++i)
    {
        work_type::execute(in.data(), out.data(), ker.data());
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
