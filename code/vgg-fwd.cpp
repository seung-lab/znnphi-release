#include "znn/bench/forward2.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<64, 64, 128, 1, 114, 1, 3>("conv2");
    // benchmark_forward<64, 128, 256, 1, 58, 1, 3>("conv3");
    // benchmark_forward<64, 256, 256, 1, 58, 1, 3>("conv4");
    // benchmark_forward<64, 256, 512, 1, 30, 1, 3>("conv5");
    // benchmark_forward<64, 512, 512, 1, 30, 1, 3>("conv6");
}
