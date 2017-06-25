#include "znn/bench/forward2.hpp"

using namespace znn::phi;

int main()
{
    //benchmark_forward<1, 16, 80, 767, 768, 3, 3>("conv1");
    //benchmark_forward<8, 80, 80, 300, 300, 3, 3>("conv2");
    //benchmark_forward<64, 80, 80, 192, 192, 3, 3>("conv3");
    //benchmark_forward<64*8, 80, 80, 76, 76, 3, 3>("conv4");
    benchmark_forward<1, 64, 64, 1, 570, 1, 3>("conv1b");
    // benchmark_forward<1, 128, 128, 1, 282, 1, 3>("conv2b");
    // benchmark_forward<1, 256, 256, 1, 138, 1, 3>("conv3b");
    // benchmark_forward<1, 512, 512, 1, 66, 1, 3>("conv4b");
    // benchmark_forward<1, 1024, 1024, 1, 30, 1, 3>("conv5b");
}
