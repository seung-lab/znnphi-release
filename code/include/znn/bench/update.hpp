#pragma once

#include "znn/layer/conv/upd/plan_advanced.hpp"
//#include "znn/layer/conv/upd3/plan2.hpp"
#include "znn/tensor/tensor.hpp"
#include <iostream>
#include <chrono>
#include <string>

namespace znn
{
namespace phi
{

template <long_t Cores, long_t HT, long_t B, long_t IFM, long_t OFM, long_t ID,
          long_t IHW, long_t KD, long_t KHW>
double benchmark_single_update(std::string const& lname = "layer")
{

    static const long_t IFM2 =
        SIMD_WIDTH * ((IFM + SIMD_WIDTH - 1) / SIMD_WIDTH);
    static const long_t OFM2 =
        SIMD_WIDTH * ((OFM + SIMD_WIDTH - 1) / SIMD_WIDTH);

    using update_type =
        upd_plan<Cores * HT, B, IFM, OFM, ID, IHW, IHW, ID - KD + 1,
                 IHW - KHW + 1, IHW - KHW + 1, KD, KHW, KHW>;

    static const long_t OD  = ID + 1 - KD;
    static const long_t OHW = IHW + 1 - KHW;

    hbw_array<float> in(one_init, B * IFM2 * ID * IHW * IHW);
    hbw_array<float> ker(one_init, IFM2 * OFM2 * KD * KHW * KHW);
    hbw_array<float> out(one_init, B * OFM2 * OD * OHW * OHW);
    hbw_array<float> bi(one_init, OFM2);
    hbw_array<float> ws(one_init, update_type::workspace_size / 4);

    // std::cout << "Benchmarking: batch" << B << " x OFM "
    //           << OFM << " x IFM " << IFM << " x IN( "
    //           << ID << " x " << IHW << " x " << IHW << " ) x KER( "
    //           << KD << " x " << KHW << " x " << KHW << " )\n";

    // std::cout << "CORES: " << Cores << " HWT: " << HT << "\n";

    kernel_launcher kl(Cores, HT, 0);

    update_type plan(&kl, in.data(), out.data(), ker.data(), ker.data(),
                     bi.data(), ws.data());

    long_t iters = 1 * Cores;

    for (long_t i = 0; i < iters; ++i)
    {
        plan.execute();
    }

    auto begin = std::chrono::high_resolution_clock::now();

    for (long_t i = 0; i < iters; ++i)
    {
        // auto begin2 = std::chrono::high_resolution_clock::now();
        plan.execute();
        // auto end2 = std::chrono::high_resolution_clock::now();
        // auto duration2 =
        // std::chrono::duration_cast<std::chrono::microseconds>
        //     (end2-begin2).count();
        // double secs  = static_cast<double>(duration2) / 1000000;
        // std::cout << secs << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    double secs = static_cast<double>(duration) / 1000000;

    std::cout << lname << "," << Cores << "," << HT << "," << (Cores * HT)
              << "," << plan.gflops() << "," << (secs / iters) << ","
              << (plan.gflops() * iters / secs) << "\n";

    // std::cout << "Secs   : " << (secs/iters) << "\n";
    // std::cout << "GFLOP/s: " << (plan.gflops()*iters/secs) << "\n\n";

    return secs / iters;
}

template <long_t B, long_t IFM, long_t OFM, long_t ID, long_t IHW, long_t KD,
          long_t KHW>
void benchmark_update(std::string const& lname = "layer")
{
    benchmark_single_update<1, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<1, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<1, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif

#if (ZNN_NUM_CORES >= 2)
    benchmark_single_update<2, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<2, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<2, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

// #if (ZNN_NUM_CORES>=6)
//     benchmark_single_update<6,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
//     benchmark_single_update<6,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #if defined(ZNN_AVX512) || defined(ZNN_KNC)
//     benchmark_single_update<6,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #endif
// #endif

#if (ZNN_NUM_CORES >= 4)
    benchmark_single_update<4, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<4, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<4, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 8)
    benchmark_single_update<8, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<8, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<8, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 16)
    benchmark_single_update<16, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<16, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<16, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 18)
    benchmark_single_update<18, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<18, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<18, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 32)
    benchmark_single_update<32, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<32, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<32, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 64)
    benchmark_single_update<64, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<64, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<64, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 70)
    benchmark_single_update<70, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<70, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<70, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 72)
    benchmark_single_update<72, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_update<72, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_update<72, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif
}
}
} // namespace znn::phi
