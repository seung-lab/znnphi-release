#pragma once

#include "znn/types.hpp"
#include "znn/intrin.hpp"

namespace znn { namespace phi { namespace upd_utils {

constexpr long_t largest_pow2_factor( long_t N )
{
    return ( N % 2 ) ? 1 :
        largest_pow2_factor(N/2) * 2;
}

constexpr bool is_power_of_2( long_t N )
{
    return ( N == 1 ) ? true :
     ( ( N % 2 ) ? false : is_power_of_2(N/2) );
}

constexpr long_t max_cpus( long_t B, long_t IFM, long_t OFM )
{
    return B * largest_pow2_factor(IFM) * largest_pow2_factor(OFM);
}

constexpr long_t get_factor( long_t C, long_t A )
{
    return ( C > A ) ? ( C / A ) : 1;
}

constexpr long_t div_factor( long_t C, long_t B, long_t IFM, long_t OFM )
{
    return get_factor( C, max_cpus(B,IFM,OFM) );
}


inline constexpr long_t smallest_prime_factor( long_t a )
{
    return ( a % 2 == 0 ) ? 2 : (
        ( a % 3 == 0 ) ? 3 : (
            ( a % 5 == 0 ) ? 5 : (
                ( a % 7 == 0 ) ? 7 : (
                    ( a % 11 == 0 ) ? 11 : a ))));
}

constexpr long_t gcd( long_t a, long_t b )
{
    return ( b == 0 ) ? a : gcd( b, a % b );
}

template< long_t B, long_t D, long_t H, long_t W >
struct bdhw
{
    static const long_t b = B;
    static const long_t d = D;
    static const long_t h = H;
    static const long_t w = W;
};

template< class Shape, long_t Threads >
struct split_bdhw
{
};

}}} // namespace znn:phi::upd_utils
