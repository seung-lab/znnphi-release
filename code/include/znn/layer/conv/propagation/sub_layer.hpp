#pragma once

#include "znn/layer/conv/propagation/full_image.hpp"
#include "znn/layer/conv/propagation/full_layer/problem.hpp"

#include <type_traits>

namespace znn
{
namespace phi
{
namespace propagation
{

template <class P>
struct sub_layer
{
private:
    using sub  = typename P::sub_problem;
    using full = typename P::original_problem;

    static_assert(P::threads == 1, "threads more than 1");

private:
    static constexpr long_t batch_size       = sub::b_len;
    static constexpr long_t in_batch_stride  = full::in_batch_stride;
    static constexpr long_t out_batch_stride = full::out_batch_stride;
    static constexpr long_t ifm_len          = full::ifm_len;
    static constexpr long_t ofm_len          = sub::ofm_len;
    static constexpr long_t ifm_stride       = full::ifm_stride;
    static constexpr long_t ofm_stride       = full::ofm_stride;
    static constexpr long_t ik_stride        = full::ik_stride;
    static constexpr long_t ok_stride        = full::ok_stride;

    template <bool First>
    using sub_task = full_image<
        First, SIMD_WIDTH, image_traits<sub::d_len, full::image_d::in_stride,
                                        full::image_d::out_stride>,
        image_traits<sub::h_len, full::image_h::in_stride,
                     full::image_h::out_stride>,
        image_traits<sub::w_len, full::image_w::in_stride,
                     full::image_w::out_stride>,
        typename full::conv_d, typename full::conv_h, typename full::conv_w>;

    static constexpr long_t in_offset =
        sub::b_from * full::in_batch_stride +
        sub::d_from * full::image_d::in_stride * full::conv_d::conv_stride +
        sub::h_from * full::image_h::in_stride * full::conv_h::conv_stride +
        sub::w_from * full::image_w::in_stride * full::conv_w::conv_stride;

    static constexpr long_t out_offset =
        sub::b_from * full::out_batch_stride +
        sub::ofm_from * full::ofm_stride +
        sub::d_from * full::image_d::out_stride +
        sub::h_from * full::image_h::out_stride +
        sub::w_from * full::image_w::out_stride;

    static constexpr long_t bias_offset = sub::ofm_from * SIMD_WIDTH;

    static constexpr long_t kernel_offset = sub::ofm_from * full::ok_stride;

private:
    static void execute_single(float const* __restrict i, float* __restrict o,
                               float const* __restrict k,
                               float const* __restrict b)
    {
        for (long_t ofm = 0; ofm < ofm_len; ++ofm)
        {
            sub_task<true>::execute(i, o + ofm * ofm_stride,
                                    k + ofm * ok_stride, b + ofm * SIMD_WIDTH);
            for (long_t ifm = 1; ifm < ifm_len; ++ifm)
            {
                sub_task<false>::execute(i + ifm * ifm_stride,
                                         o + ofm * ofm_stride,
                                         k + ifm * ik_stride + ofm * ok_stride,
                                         b + ofm * SIMD_WIDTH);
            }
        }
    }

public:
    static void execute(float const* __restrict i, float* __restrict o,
                        float const* __restrict k, float const* __restrict b)
    {
        for (long_t bs = 0; bs < batch_size; ++bs)
        {
            execute_single(i + in_offset + bs * in_batch_stride,
                           o + out_offset + bs * out_batch_stride,
                           k + kernel_offset, b + bias_offset);
        }
    }

    static long_t flops()
    {
        return batch_size * ifm_len * ofm_len * SIMD_WIDTH * SIMD_WIDTH *
               sub::d_len * sub::h_len * sub::w_len * full::conv_d::size *
               full::conv_h::size * full::conv_w::size * 2;
    }
};

template <>
struct sub_layer<null_problem_t>
{
    static void execute(float const* __restrict, float* __restrict,
                        float const* __restrict, float const* __restrict)
    {
    }

    static long_t flops() { return 0; }
};

} // namespace propagation
} // namespace phi
} // namespace znn
