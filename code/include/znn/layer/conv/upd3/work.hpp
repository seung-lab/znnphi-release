#pragma once

#include "znn/intrin.hpp"

#if (SIMD_WIDTH > 8)

#include "znn/layer/conv/upd3/work_knl.hpp"

#else

#include "znn/layer/conv/upd3/work_xeon.hpp"

#endif
