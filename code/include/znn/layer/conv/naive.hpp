#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include <type_traits>
#include <iostream>
#include <chrono>

namespace znn { namespace phi {

namespace detail {

template< typename... T >
struct and_: std::true_type {};

template<typename First, typename... Rest>
struct and_< First, Rest...>
    : std::integral_constant< bool,
                              First::value && and_<Rest...>::value
                              >
{};


template< long_t... >
struct index_calculator;

template< long_t I >
struct index_calculator< I >
{
    static constexpr long_t size   = I;
    static constexpr long_t stride = 1;

    static constexpr long_t get( long_t n )
    {
        return n;
    }
};

template< long_t I, long_t... Is >
struct index_calculator< I, Is...>
{
private:
    using inner_type = index_calculator<Is...>;

public:
    static constexpr long_t size = I;
    static constexpr long_t stride =
        inner_type::size * inner_type::stride;

    template< typename... T >
    static constexpr
    typename std::enable_if<
        and_<std::integral_constant<bool,sizeof...(T)==sizeof...(Is)>,
        std::is_same<T, long_t>...>::value, long_t
        >::type
    get( long_t n, T... ts )
    {
        return n * stride + inner_type::get(ts...);
    }

};

} // namespace detail


template< typename T, long_t... Is >
struct static_tensor
{
private:
    using indices = detail::index_calculator<Is...>;
    static constexpr long_t num_elements = indices::size * indices::stride;
    host_array<T> data = host_array<T>(one_init,num_elements);

public:
    template< typename... Y >
    typename std::enable_if<
        detail::and_<std::integral_constant<bool,sizeof...(Y)==sizeof...(Is)>,
        std::is_same<Y, long_t>...>::value, T &
        >::type
    operator()( Y... ts )
    {
        return *(data.data() + indices::get(ts...));
    }
};


inline void bench_very_naive( long_t B,
                              long_t IFM, long_t OFM,
                              long_t ID , long_t IHW,
                              long_t KD , long_t KHW,
                              long_t iters )
{
    long_t OD  = ID  + 1 - KD ;
    long_t OHW = IHW + 1 - KHW;

    host_tensor<float,5> in ( one_init, B, IFM, ID, IHW, IHW );
    host_tensor<float,5> out( one_init, B, OFM, OD, OHW, OHW );
    host_tensor<float,5> ks ( one_init, OFM, IFM, KD, KHW, KHW );
    host_tensor<float,1> bs ( one_init, OFM );

    double flops = B * IFM * OFM * OD * OHW * OHW * KD * KHW * KHW * 2;
    double gflops = flops * iters / 1000000000;

    auto begin = std::chrono::high_resolution_clock::now();

    for ( long_t iter = 0; iter < iters; ++ iter )
    {
        for ( long_t b = 0; b < B; ++b )
            for ( long_t oi = 0; oi < OFM; ++oi )
                for ( long_t ox = 0; ox < OD; ++ox )
                    for ( long_t oy = 0; oy < OHW; ++oy )
                        for ( long_t oz = 0; oz < OHW; ++oz )
                        {
                            out[b][oi][ox][oy][oz] = bs[oi];

                            for ( long_t ii = 0; ii < IFM; ++ii )
                                for ( long_t kx = 0; kx < KD; ++kx )
                                    for ( long_t ky = 0; ky < KHW; ++ky )
                                        for ( long_t kz = 0; kz < KHW; ++kz )

                                            out[b][oi][ox][oy][oz] +=
                                                in[b][ii][ox+kx][oy+ky][oz+kz] *
                                                ks[oi][ii][kx][ky][kz];
                        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (end-begin).count();

    double secs  = static_cast<double>(duration) / 1000000;

    std::cout << "Secs   : " << (secs/iters) << "\n";
    std::cout << "GFLOP/s: " << (gflops/secs) << "\n\n";

};


template< long_t B,
          long_t IFM, long_t OFM,
          long_t ID , long_t IHW,
          long_t KD , long_t KHW >
inline void bench_naive(long_t iters )
{
    static constexpr long_t OD  = ID  + 1 - KD ;
    static constexpr long_t OHW = IHW + 1 - KHW;

    static_tensor<float,B,IFM,ID,IHW,IHW> in;
    static_tensor<float,B,OFM,OD,OHW,OHW> out;
    static_tensor<float,OFM,IFM,KD,KHW,KHW> ks;
    static_tensor<float,OFM> bs;

    double flops = B * IFM * OFM * OD * OHW * OHW * KD * KHW * KHW * 2;
    double gflops = flops * iters / 1000000000;

    auto begin = std::chrono::high_resolution_clock::now();

    for ( long_t iter = 0; iter < iters; ++ iter )
    {
        for ( long_t b = 0; b < B; ++b )
            for ( long_t oi = 0; oi < OFM; ++oi )
                for ( long_t ox = 0; ox < OD; ++ox )
                    for ( long_t oy = 0; oy < OHW; ++oy )
                        for ( long_t oz = 0; oz < OHW; ++oz )
                        {
                            out(b,oi,ox,oy,oz) = bs(oi);

                            for ( long_t ii = 0; ii < IFM; ++ii )
                                for ( long_t kx = 0; kx < KD; ++kx )
                                    for ( long_t ky = 0; ky < KHW; ++ky )
                                        for ( long_t kz = 0; kz < KHW; ++kz )

                                            out(b,oi,ox,oy,oz) +=
                                                in(b,ii,ox+kx,oy+ky,oz+kz) *
                                                ks(oi,ii,kx,ky,kz);
                        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (end-begin).count();

    double secs  = static_cast<double>(duration) / 1000000;

    std::cout << "Secs   : " << (secs/iters) << "\n";
    std::cout << "GFLOP/s: " << (gflops/secs) << "\n\n";

};



}} // namespace znn::phi
