#pragma once

#include "znn/intrin.hpp"
#include "znn/layer/conv/fwd4/plan.hpp"
#include "znn/tensor/tensor.hpp"
#include <chrono>
#include <iostream>
#include <string>

namespace znn
{
namespace phi
{

template <long_t Cores, long_t HT, long_t B, long_t IFM, long_t OFM, long_t ID,
          long_t IHW, long_t KD, long_t KHW, long_t PADD = 0, long_t PADHW = 0>
double benchmark_single_forward(std::string const& lname = "layer")
{
    static const long_t IFM2 =
        SIMD_WIDTH * ((IFM + SIMD_WIDTH - 1) / SIMD_WIDTH);
    static const long_t OFM2 =
        SIMD_WIDTH * ((OFM + SIMD_WIDTH - 1) / SIMD_WIDTH);

    static const long_t OD  = ID + 1 - KD + 2 * PADD;
    static const long_t OHW = IHW + 1 - KHW + 2 * PADHW;

    hbw_array<float> in(one_init, B * IFM2 * ID * IHW * (IHW + PADHW * 2));
    hbw_array<float> ker(one_init, IFM2 * OFM2 * KD * KHW * KHW);
    hbw_array<float> out(one_init, B * OFM2 * OD * OHW * OHW);
    hbw_array<float> bi(one_init, OFM2 * SIMD_WIDTH);

    // std::cout << "Benchmarking: batch" << B << " x OFM "
    //           << OFM << " x IFM " << IFM << " x IN( "
    //           << ID << " x " << IHW << " x " << IHW << " ) x KER( "
    //           << KD << " x " << KHW << " x " << KHW << " )\n";

    // std::cout << "CORES: " << Cores << " HWT: " << HT << "\n";

    kernel_launcher kl(Cores, HT, 0);

    fwd_plan<Cores * HT, B, OFM, IFM, ID, IHW, IHW, OD, OHW, OHW, KD, KHW, KHW>
        plan(&kl, in.data(), out.data(), ker.data(), bi.data());

    long_t iters = 5 * Cores;
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

    // std::cout << "Secs   : " << (secs/iters) << "\n";
    // std::cout << "GFLOP/s: " << (plan.gflops()*iters/secs) << "\n\n";

    std::cout << lname << "," << Cores << "," << HT << "," << (Cores * HT)
              << "," << plan.gflops() << "," << (secs / iters) << ","
              << (plan.gflops() * iters / secs) << "\n";

    return secs / iters;
}

template <long_t B, long_t IFM, long_t OFM, long_t ID, long_t IHW, long_t KD,
          long_t KHW, long_t PADD = 0, long_t PADHW = 0>
void benchmark_forward(std::string const& lname = "layer")
{
    // std::cout << "Benchmarked system has " << ZNN_NUM_CORES
    //           << " cores ";

    // #if defined(ZNN_AVX)
    //     std::cout << "and AVX\n";
    // #elif defined(ZNN_AVX2)
    //     std::cout << "and AVX2\n";
    // #elif defined(ZNN_AVX512)
    //     std::cout << "and AVX512\n";
    // #endif

    // benchmark_single_forward<60,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
    // benchmark_single_forward<60,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
    // benchmark_single_forward<60,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );

    // benchmark_single_forward<63,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
    // benchmark_single_forward<63,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
    // benchmark_single_forward<63,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );

    // benchmark_single_forward<32,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
    // benchmark_single_forward<32,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
    // benchmark_single_forward<32,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );

    //#if 0

    benchmark_single_forward<1, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<1, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<1, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif

#if (ZNN_NUM_CORES >= 2)
    benchmark_single_forward<2, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<2, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<2, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

// #if (ZNN_NUM_CORES>=3)
//     benchmark_single_forward<3,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
//     benchmark_single_forward<3,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #if defined(ZNN_AVX512) || defined(ZNN_KNC)
//     benchmark_single_forward<3,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #endif
// #endif

#if (ZNN_NUM_CORES >= 4)
    benchmark_single_forward<4, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<4, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<4, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

// #if (ZNN_NUM_CORES>=6)
//     benchmark_single_forward<6,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
//     benchmark_single_forward<6,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #if defined(ZNN_AVX512) || defined(ZNN_KNC)
//     benchmark_single_forward<6,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #endif
// #endif

#if (ZNN_NUM_CORES >= 10)
    benchmark_single_forward<10, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<10, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<10, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 16)
    benchmark_single_forward<16, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<16, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<16, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if ((ZNN_NUM_CORES >= 18) && ((ZNN_NUM_CORES % 18) == 0))
    benchmark_single_forward<18, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<18, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<18, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if (ZNN_NUM_CORES >= 32)
    benchmark_single_forward<32, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<32, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<32, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

#if ((ZNN_NUM_CORES >= 36) && ((ZNN_NUM_CORES % 18) == 0))
    benchmark_single_forward<36, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<36, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<36, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

// // #if (ZNN_NUM_CORES>=54)
// //     benchmark_single_forward<54,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// //     benchmark_single_forward<54,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// // #if defined(ZNN_AVX512) || defined(ZNN_KNC)
// //     benchmark_single_forward<54,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// // #endif
// // #endif

// #if (ZNN_NUM_CORES>=60)
//     benchmark_single_forward<60,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
//     benchmark_single_forward<60,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #if defined(ZNN_AVX512) || defined(ZNN_KNC)
//     benchmark_single_forward<60,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #endif
// #endif

// #if (ZNN_NUM_CORES>=63)
//     benchmark_single_forward<63,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
//     benchmark_single_forward<63,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #if defined(ZNN_AVX512) || defined(ZNN_KNC)
//     benchmark_single_forward<63,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #endif
// #endif

#if (ZNN_NUM_CORES >= 64)
    benchmark_single_forward<64, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<64, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<64, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

// #if ((ZNN_NUM_CORES>=70) && ((ZNN_NUM_CORES%18)==0))
//     benchmark_single_forward<70,1,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
//     benchmark_single_forward<70,2,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #if defined(ZNN_AVX512) || defined(ZNN_KNC)
//     benchmark_single_forward<70,4,B,IFM,OFM,ID,IHW,KD,KHW>( lname );
// #endif
// #endif

#if ((ZNN_NUM_CORES >= 72) && ((ZNN_NUM_CORES % 18) == 0))
    benchmark_single_forward<72, 1, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
    benchmark_single_forward<72, 2, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#if defined(ZNN_AVX512) || defined(ZNN_KNC)
    benchmark_single_forward<72, 4, B, IFM, OFM, ID, IHW, KD, KHW>(lname);
#endif
#endif

    //#endif
}
}
} // namespace znn::phi
