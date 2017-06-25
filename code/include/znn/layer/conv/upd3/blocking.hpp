#pragma once

#include "znn/types.hpp"
#include "znn/intrin.hpp"

namespace znn { namespace phi {

#if (SIMD_WIDTH>8)

#define ZNN_BLOCK_W_MAX (SIMD_MAX_BLOCK/2)

template< long_t D, long_t H, long_t W >
struct upd_blocking_t
{
private:
    static constexpr long_t to_pow_of_2_h( long_t k, long_t n )
    {
        return ( 2*k > n ) ? k : to_pow_of_2_h( 2*k, n );
    };

    static constexpr long_t to_pow_of_2( long_t n )
    {
        return to_pow_of_2_h( 1, n );
    };

public:
    static const long_t w = ( W <= SIMD_W_BLOCK ) ? W : 1;
    static const long_t width = w;
private:
    static const long_t rem_for_h = SIMD_MAX_BLOCK / w;
public:
    static const long_t h = 1;//( H <= rem_for_h ) ? H : 1;
    static const long_t height = h;
private:
    static const long_t rem_for_d = rem_for_h / h;
public:
    static const long_t d = 1;// ( D <= rem_for_d ) ? D : 1;
    static const long_t depth = d;
    static const long_t f = to_pow_of_2(rem_for_d / d);
};

#else


template< long_t D, long_t H, long_t W >
struct upd_blocking_t
{
private:
    static constexpr long_t to_pow_of_2_h( long_t k, long_t n )
    {
        return ( 2*k > n ) ? k : to_pow_of_2_h( 2*k, n );
    };

    static constexpr long_t to_pow_of_2( long_t n )
    {
        return to_pow_of_2_h( 1, n );
    };

public:
    static const long_t w = ( W <= 6 ) ? W : 1;
    static const long_t width = w;
private:
    static const long_t rem_for_h = 6 / w;
public:
    static const long_t h = 1;//( H <= rem_for_h ) ? H : 1;
    static const long_t height = h;
private:
    static const long_t rem_for_d = rem_for_h / h;
public:
    static const long_t d = 1;//  ( D <= rem_for_d ) ? D : 1;
    static const long_t depth = d;
    static const long_t f = to_pow_of_2(rem_for_d / d);
};

#endif

}} // namespace znn:phi
