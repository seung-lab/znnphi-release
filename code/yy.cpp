#include "znn/layer/conv/propagation/full_layer.hpp"
#include "znn/layer/conv/propagation/traits.hpp"
#include "znn/tensor/tensor.hpp"

using namespace znn::phi;

template <long_t BS,                                   // batch size
          long_t IFM, long_t OFM,                      // ifm/ofm sets
          long_t OD, long_t OH, long_t OW,             // output size
          long_t KD, long_t KH, long_t KW,             // kernel size
          long_t SD = 1, long_t SH = 1, long_t SW = 1> // conv stride
inline void
test()
{
    using namespace propagation;

    static constexpr long_t ID = (OD - 1) * SD + KD;
    static constexpr long_t IH = (OH - 1) * SH + KH;
    static constexpr long_t IW = (OW - 1) * SW + KW;

    using orig_prob = original_problem_t<
        BS,                                           // batch size
        IFM * ID * IH * IW * SIMD_WIDTH,              // in batch stride
        OFM * OD * OH * OW * SIMD_WIDTH,              // out batch stride
        IFM, OFM,                                     // ifm / ofm
        ID * IH * IW * SIMD_WIDTH,                    // ifm stride
        OD * OH * OW * SIMD_WIDTH,                    // ofm stride
        KD * KH * KW * SIMD_WIDTH * SIMD_WIDTH,       // kernel in stride
        KD * KH * KW * SIMD_WIDTH * SIMD_WIDTH * IFM, // kernel out stride
        image_traits<OD, IH * IW * SIMD_WIDTH, OH * OW * SIMD_WIDTH>,
        image_traits<OH, IW * SIMD_WIDTH, OW * SIMD_WIDTH>,
        image_traits<OW, SIMD_WIDTH, SIMD_WIDTH>, conv_traits<KD, KW * KH, SD>,
        conv_traits<KH, KW, SH>, conv_traits<KW, 1, SW>>;

    full_layer<144,orig_prob> fl;

    std::cout << fl.flops() << "\n";
}

int main()
{
    print_compiler_version();

    test<12,10,12,1,112,112,1,3,3>();
}
