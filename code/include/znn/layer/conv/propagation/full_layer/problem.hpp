#pragma once

#include "znn/intrin.hpp"
#include "znn/layer/conv/propagation/full_image.hpp"
#include "znn/layer/conv/propagation/traits.hpp"
#include "znn/types.hpp"

namespace znn
{
namespace phi
{
namespace propagation
{

template <long_t BFrom, long_t BN, // which batches
          long_t OFrom, long_t ON, // which outputs
          long_t DFrom, long_t DN, // which depths
          long_t HFrom, long_t HN, // which height
          long_t WFrom, long_t WN> // which width
struct sub_problem_t
{
    static constexpr long_t b_from   = BFrom;
    static constexpr long_t b_len    = BN;
    static constexpr long_t ofm_from = OFrom;
    static constexpr long_t ofm_len  = ON;
    static constexpr long_t d_from   = DFrom;
    static constexpr long_t d_len    = DN;
    static constexpr long_t h_from   = HFrom;
    static constexpr long_t h_len    = HN;
    static constexpr long_t w_from   = WFrom;
    static constexpr long_t w_len    = WN;
};

template <long_t BS,                        // batch size
          long_t IBStride, long_t OBStride, // I/O batch strides
          long_t IFM, long_t OFM, // number of I/O channels in S multiples
          long_t IFMStride, long_t OFMStride, // ifm/ofm strides,
          long_t IKStride, long_t OKStride,   // kernel strides
          class ID, class IH, class IW,       // image traits
          class CD, class CH, class CW>       // convolution traits
struct original_problem_t
{
    static constexpr long_t batch_size       = BS;
    static constexpr long_t in_batch_stride  = IBStride;
    static constexpr long_t out_batch_stride = OBStride;
    static constexpr long_t ifm_len          = IFM;
    static constexpr long_t ofm_len          = OFM;
    static constexpr long_t ifm_stride       = IFMStride;
    static constexpr long_t ofm_stride       = OFMStride;
    static constexpr long_t ik_stride        = IKStride;
    static constexpr long_t ok_stride        = OKStride;

    using image_d = ID;
    using image_h = IH;
    using image_w = IW;
    using conv_d  = CD;
    using conv_h  = CH;
    using conv_w  = CW;
};

template <long_t Threads,        // available threads
          class OriginalProblem, // original problem
          class SubProblem>      // current problem size
struct problem_t
{
    static constexpr long_t threads = Threads;
    using original_problem          = OriginalProblem;
    using sub_problem               = SubProblem;
};

struct null_problem_t
{
};

} // namespace propagation
} // namespace phi
} // namespace znn
