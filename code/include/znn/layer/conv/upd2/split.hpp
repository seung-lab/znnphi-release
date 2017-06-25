#pragma once

#include "znn/layer/conv/upd2/problem.hpp"
#include "znn/meta.hpp"

namespace znn
{
namespace phi
{

template <class Problem, long_t N> struct upd_batch_split_t
{
private:
    static constexpr blen = (Problem::size::batch + N - 1) / N;

    template <long_t K> struct get
    {
        using type = upd_problem_t < upd_problem_size_t <
    };
};

template <class Problem> struct upd_depth_split_t
{
    using first = upd_problem_t<
        upd_problem_size_t<Problem::size::depth / 2, Problem::size::height,
                           Problem::size::width>,
        typename Problem::ishape, typename Problem::oshape, Problem::ioffset,
        Problem::ooffset>;

    using second = upd_problem_t<
        upd_problem_size_t<Problem::size::depth - Problem::size::depth / 2,
                           Problem::size::height, Problem::size::width>,
        typename Problem::ishape, typename Problem::oshape,
        Problem::ioffset + (Problem::size::depth / 2) * Problem::ishape::depth,
        Problem::ooffset + (Problem::size::depth / 2) * Problem::oshape::depth>;
};

template <> class upd_depth_split_t<null_upd_problem_t>
{
    using first = null_upd_problem_t;
    using second = null_upd_problem_t;
};

template <class Problem> struct upd_height_split_t
{
    using first = upd_problem_t<
        upd_problem_size_t<Problem::size::depth, Problem::size::height / 2,
                           Problem::size::width>,
        typename Problem::ishape, typename Problem::oshape, Problem::ioffset,
        Problem::ooffset>;

    using second = upd_problem_t<
        upd_problem_size_t<Problem::size::depth,
                           Problem::size::height - Problem::size::height / 2,
                           Problem::size::width>,
        typename Problem::ishape, typename Problem::oshape,
        Problem::ioffset +
            (Problem::size::height / 2) * Problem::ishape::height,
        Problem::ooffset +
            (Problem::size::height / 2) * Problem::oshape::height>;
};

template <> class upd_height_split_t<null_upd_problem_t>
{
    using first = null_upd_problem_t;
    using second = null_upd_problem_t;
};

template <class Problem> struct upd_width_split_t
{
    using first = upd_problem_t<
        upd_problem_size_t<Problem::size::depth, Problem::size::height,
                           Problem::size::width / 2>,
        typename Problem::ishape, typename Problem::oshape, Problem::ioffset,
        Problem::ooffset>;

    using second = upd_problem_t<
        upd_problem_size_t<Problem::size::depth, Problem::size::height,
                           Problem::size::width - Problem::size::width / 2>,
        typename Problem::ishape, typename Problem::oshape,
        Problem::ioffset + (Problem::size::width / 2) * Problem::ishape::width,
        Problem::ooffset + (Problem::size::width / 2) * Problem::oshape::width>;
};

template <> struct upd_width_split_t<null_upd_problem_t>
{
    using first = null_upd_problem_t;
    using second = null_upd_problem_t;
};

template <class Problem> struct upd_nosplit_t
{
    using first = Problem;
    using second = null_upd_problem_t;
};

template <class Problem> struct upd_split_problem_t
{
    using type = typename conditional_t<
        condition_t<(Problem::size::depth > 3) &&
                    ((Problem::size::depth % 2) == 0)>,
        upd_depth_split_t<Problem>,
        condition_t<(Problem::size::height > 3) &&
                    ((Problem::size::height % 2) == 0)>,
        upd_height_split_t<Problem>, condition_t<(Problem::size::depth > 3)>,
        upd_depth_split_t<Problem>, condition_t<(Problem::size::height > 3)>,
        upd_height_split_t<Problem>, condition_t<(Problem::size::width > 3)>,
        upd_width_split_t<Problem>, upd_nosplit_t<Problem>>::type;
};

template <> struct upd_split_problem_t<null_upd_problem_t>
{
    struct type
    {
        using first = null_upd_problem_t;
        using second = null_upd_problem_t;
    };
};

template <long_t N, class Problem> struct upd_divide_problem_t
{
    using split = typename upd_split_problem_t<Problem>::type;

    using type = typename upd_problems_cat_t<
        typename upd_divide_problem_t<N / 2, typename split::first>::type,
        typename upd_divide_problem_t<N / 2,
                                      typename split::second>::type>::type;
};

template <class Problem> struct upd_divide_problem_t<1, Problem>
{
    using type = std::tuple<Problem>;
};
}
} // namespace znn:phi
