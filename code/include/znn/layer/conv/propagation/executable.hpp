#pragma once

#include "znn/types.hpp"

#include <vector>

namespace znn
{
namespace phi
{
namespace propagation
{

typedef void (*executable_t)(float const* __restrict, float* __restrict,
                             float const* __restrict, float const* __restrict);

using exec_vector = std::vector<std::vector<executable_t>>;

} // namespace propagation
} // namespace phi
} // namespace znn
