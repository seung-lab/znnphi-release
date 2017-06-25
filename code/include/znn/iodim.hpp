#pragma once

#include "znn/types.hpp"

namespace znn { namespace phi {

template< long_t N, long_t IS, long_t OS >
struct iodim_t
{
    static const long_t n  = N ;
    static const long_t is = IS;
    static const long_t os = OS;
};


template< long_t N, long_t S >
struct dim_t
{
    static const long_t n = N;
    static const long_t s = S;
};


}} // namespace znn:phi
