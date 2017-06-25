#pragma once

#include "znn/types.hpp"

namespace znn
{
namespace phi
{
namespace update
{

template <long_t BFrom, long_t BN, // which batches
          long_t DFrom, long_t DN, // which depths
          long_t HFrom, long_t HN, // which heights
          long_t WFrom, long_t WN> // which widths
struct sub_problem_t
{
    static constexpr long_t b_from = BFrom;
    static constexpr long_t b_len  = BN;
    static constexpr long_t d_from = DFrom;
    static constexpr long_t d_len  = DN;
    static constexpr long_t h_from = HFrom;
    static constexpr long_t h_len  = HN;
    static constexpr long_t w_from = WFrom;
    static constexpr long_t w_len  = WN;
};

template <long_t BS,                        // batch size
          long_t IBStride, long_t OBStride, // I/O batch strides
          class ID, class IH, class IW,     // image traits
          class CD, class CH, class CW>     // convolution traits
struct original_problem_t
{
    static constexpr long_t batch_size       = BS;
    static constexpr long_t in_batch_stride  = IBStride;
    static constexpr long_t out_batch_stride = OBStride;

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

} // namespace update
} // namespace phi
} // namespace znn
