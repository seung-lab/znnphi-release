#pragma once

#include "znn/types.hpp"
#include <memory>

namespace znn
{
namespace phi
{

class task
{
public:
    virtual void                  execute() const = 0;
    virtual std::unique_ptr<task> offset_copy(long_t, long_t, long_t) const = 0;
    virtual long_t                flops() const = 0;
    virtual void                  prefetch(float*) {}

    double gflops() const { return static_cast<double>(flops()) / 1000000000; }
};
}
} // namespace znn:phi
