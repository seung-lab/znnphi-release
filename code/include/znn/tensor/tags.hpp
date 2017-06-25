#pragma once

namespace znn
{
namespace phi
{
namespace detail
{
namespace tensor
{

struct random_initialize_tag
{
};

struct host_tag
{
};
struct device_tag
{
};
struct hbw_tag
{
};

struct one_init_tag
{
};
struct zero_init_tag
{
};
}
}
}
} // namespace znn::fwd::detail::tensor
