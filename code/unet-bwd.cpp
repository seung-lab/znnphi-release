#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<1,64,64,1,574,1,3>("conv1b");

    benchmark_forward<1,64,128,1,288,1,3>("conv2a");
    benchmark_forward<1,128,128,1,286,1,3>("conv2b");

    benchmark_forward<1,128,256,1,144,1,3>("conv3a");
    benchmark_forward<1,256,256,1,142,1,3>("conv3b");

    benchmark_forward<1,256,512,1,72,1,3>("conv4a");
    benchmark_forward<1,512,512,1,70,1,3>("conv4b");

    benchmark_forward<1,512,1024,1,36,1,3>("conv5a");
    benchmark_forward<1,1024,1024,1,34,1,3>("conv5b");

}
