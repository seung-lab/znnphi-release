#pragma once

#include "znn/types.hpp"

namespace znn { namespace phi {


template< long_t DFrom, long_t DTo,
          long_t HFrom, long_t HTo >
struct fwd_offset_t
{
    static const long_t d_from = DFrom ;
    static const long_t d_to   = DTo   ;
    static const long_t h_from = HFrom;
    static const long_t h_to   = HTo  ;
};

struct null_fwd_problem_t
{
    static const long_t ioffset = 0;
    static const long_t ooffset = 0;
    static const long_t koffset = 0;
    static const long_t boffset = 0;

    using offsets = fwd_offset_t<0,0,0,0>;
};


template< long_t Threads,
          class  ProblemSize,
          long_t InputOffset,
          long_t OutputOffset,
          long_t KernelOffset,
          long_t BiasOffset,
          class  Offsets,
          class  Shapes >
struct fwd_problem_t
{
    static const long_t threads = Threads;
    static const long_t ioffset = InputOffset;
    static const long_t ooffset = OutputOffset;
    static const long_t koffset = KernelOffset;
    static const long_t boffset = BiasOffset;

    using size   = ProblemSize;
    using shapes = Shapes;

    using offsets = Offsets;
};

template< class ProblemSize,
          class Shapes >
struct fwd_serial_problem_t
{
    using size   = ProblemSize;
    using shapes = Shapes;
};

template< class Problem >
struct extract_serial_problem
{
    using type = fwd_serial_problem_t< typename Problem::size,
                                       typename Problem::shapes >;
};

template<>
struct extract_serial_problem<null_fwd_problem_t>
{
    using type = null_fwd_problem_t;
};

template< long_t B,
          long_t I,
          long_t F,
          long_t D,
          long_t H,
          long_t W >
struct fwd_problem_size_t
{
    static const long_t batch    = B;
    static const long_t ofm_sets = F;
    static const long_t depth    = D;
    static const long_t height   = H;
    static const long_t width    = W;
    static const long_t ifm      = I;
};

template< class I, class O, class W >
struct fwd_shapes_t
{
    using input  = I;
    using output = O;
    using weight = W;
};

template< long_t B, long_t F, long_t D, long_t H, long_t W >
struct fwd_ioshape_t
{
    static const long_t batch  = B;
    static const long_t fm_set = F;
    static const long_t depth  = D;
    static const long_t height = H;
    static const long_t width  = W;
};

template< long_t O,
          long_t D,
          long_t H,
          long_t W >
struct fwd_wshape_t
{
    static const long_t output = O;
    static const long_t depth  = D;
    static const long_t height = H;
    static const long_t width  = W;
    static const long_t input  = D*H*W*SIMD_WIDTH*SIMD_WIDTH;
};



}} // namespace znn:phi
