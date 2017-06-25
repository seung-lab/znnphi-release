#include "znn/bench/forward2.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<1, 64, 64, 1, 1024, 1, 3>("MX");
}
