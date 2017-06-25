#include "znn/layer/conv/propagation/full_layer/schedule.hpp"

using namespace znn::phi::propagation;

int main()
{

    using sub  = sub_problem_t<0,64,0,64,0,1,0,112,0,112>;
    using prob = problem_t<72,int,sub>;


    scheduler<prob>::schedule(0);
}
