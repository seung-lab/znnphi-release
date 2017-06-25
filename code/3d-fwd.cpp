#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<32, 64, 128, 18, 58, 3, 3>("conv2a");
    benchmark_forward<32, 128, 256, 10, 30, 3, 3>("conv3a");
    benchmark_forward<32, 256, 256, 10, 30, 3, 3>("conv3b");
    benchmark_forward<32, 256, 512, 6, 14, 3, 3>("conv4a");
    benchmark_forward<32, 512, 512, 6, 14, 3, 3>("conv4b");
}
