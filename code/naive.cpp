#include "znn/layer/conv/naive.hpp"

#include <iostream>

using namespace znn::phi;

int main()
{
    //std::cout << index_calculator<2,2,3,3>::get(1,1,1,1) << "\n";

    bench_very_naive( 1, 64,64, 1, 114, 1, 3, 1 );
    bench_very_naive( 1, 64,64, 1, 114, 1, 3, 10 );

    bench_naive< 1, 64,64, 1, 114, 1, 3 >( 1 );
    bench_naive< 1, 64,64, 1, 114, 1, 3 >( 10 );


}
