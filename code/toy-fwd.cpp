#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<64, 48, 96, 1, 114, 1, 3>("2D1");
    benchmark_forward<64, 48, 96, 1, 58, 1, 5>("2D2");
    benchmark_forward<64, 48, 96, 1, 58, 1, 11>("2D3");

    benchmark_forward<32, 48, 96, 10, 58, 2, 3>("3D1");
    benchmark_forward<32, 48, 96, 10, 58, 3, 5>("3D2");
}
