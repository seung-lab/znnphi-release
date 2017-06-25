#pragma once

#include "znn/intrin.hpp"

#if (SIMD_WIDTH > 8)

#include "work_knl.hpp"

#else

#include "work_xeon.hpp"

#endif
