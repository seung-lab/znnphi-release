#pragma once

#include "znn/types.hpp"

namespace znn { namespace phi {


template< long_t Size, long_t InStride, long_t OutStride >
struct dimension
{
    static const long_t s  = Size;
    static const long_t is = InStride;
    static const long_t os = OutStride;

    static const long_t d = Size;
    static const long_t h = InStride;
    static const long_t w = OutStride;
};

template< long_t B , long_t F , long_t D , long_t H , long_t W,
          long_t BS = F*D*H*W,
          long_t FS = D*H*W,
          long_t DS = H*W,
          long_t HS = W,
          long_t WS = 1
          >
struct shape
{
    static const long_t b = B;
    static const long_t f = F;
    static const long_t d = D;
    static const long_t h = H;
    static const long_t w = W;

    static const long_t bs = BS;
    static const long_t fs = FS;
    static const long_t ds = DS;
    static const long_t hs = HS;
    static const long_t ws = WS;
};

template< long_t S, long_t Stride, long_t Dilation >
struct conv_traits
{
    static const long_t s        = S       ;
    static const long_t stride   = Stride  ;
    static const long_t dilation = Dilation;
};

template< long_t F, long_t D, long_t H, long_t W >
struct kernel_blocking
{
    static const long_t f = F;
    static const long_t d = D;
    static const long_t h = H;
    static const long_t w = W;
};

template< long_t N, long_t IS, long_t OS >
struct iodim
{
    static const long_t n  = N ;
    static const long_t is = IS;
    static const long_t os = OS;
};

}} // namespace znn:phi
