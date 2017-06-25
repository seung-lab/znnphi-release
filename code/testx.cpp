#include "znn/layer/conv/update/full_layer/split.hpp"
#include "znn/layer/conv/update/full_layer/problems_printer.hpp"

using namespace znn::phi::update;


int main()
{
    using pt1 = problem_t<16, void, sub_problem_t<0,9,0,10,0,17,0,16>>;

    using x = split_problem_t<pt1>;


    problems_printer<x>::print();
}
