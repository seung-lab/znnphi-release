#pragma once

#include "znn/layer/conv/fwd4/problem.hpp"
#include "znn/meta.hpp"

namespace znn
{
namespace phi
{

template <class Problem, long_t N>
struct fwd_batch_splitn_t
{
private:
    using size   = typename Problem::size;
    using shapes = typename Problem::shapes;

    using part_size =
        fwd_problem_size_t<size::batch / N, size::ifm, size::ofm_sets,
                           size::depth, size::height, size::width>;

    using rest_size =
        fwd_problem_size_t<size::batch % N, size::ifm, size::ofm_sets,
                           size::depth, size::height, size::width>;

public:
    template <long_t K>
    struct part
    {
        using type = fwd_problem_t<
            Problem::threads / N, part_size,
            Problem::ioffset + shapes::input::batch*(size::batch / N) * K,
            Problem::ooffset + shapes::output::batch*(size::batch / N) * K,
            Problem::koffset, Problem::boffset, shapes>;
    };

    using rest = typename std::conditional<
        (size::batch % N) == 0, null_fwd_problem_t,
        fwd_problem_t<Problem::threads, rest_size,
                      Problem::ioffset + shapes::input::batch*(
                                             size::batch - (size::batch % N)),
                      Problem::ooffset + shapes::output::batch*(
                                             size::batch - (size::batch % N)),
                      Problem::koffset, Problem::boffset, shapes>>::type;
};

template <class Problem, long_t N>
struct fwd_ofm_splitn_t
{
private:
    using size   = typename Problem::size;
    using shapes = typename Problem::shapes;

    using part_size =
        fwd_problem_size_t<size::batch, size::ifm, size::ofm_sets / N,
                           size::depth, size::height, size::width>;

    using rest_size =
        fwd_problem_size_t<size::batch, size::ifm, size::ofm_sets % N,
                           size::depth, size::height, size::width>;

public:
    template <long_t K>
    struct part
    {

        using type = fwd_problem_t<
            Problem::threads / N, part_size, Problem::ioffset,
            Problem::ooffset + shapes::output::fm_set*(size::ofm_sets / N) * K,
            Problem::koffset + shapes::weight::output*(size::ofm_sets / N) * K,
            Problem::boffset + (size::ofm_sets / N) * K, shapes>;
    };

    using rest = typename std::conditional<
        (size::ofm_sets % N) == 0, null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads, rest_size, Problem::ioffset,
            Problem::ooffset +
                shapes::output::fm_set*(size::ofm_sets - (size::ofm_sets % N)),
            Problem::koffset +
                shapes::weight::output*(size::ofm_sets - (size::ofm_sets % N)),
            Problem::boffset + size::ofm_sets - (size::ofm_sets % N),
            shapes>>::type;
};

template <class Problem, long_t N>
struct fwd_depth_splitn_t
{
private:
    using size   = typename Problem::size;
    using shapes = typename Problem::shapes;

    static const long_t dlen  = size::depth / N;
    static const long_t drest = size::depth % N;

    using part_size = fwd_problem_size_t<size::batch, size::ifm, size::ofm_sets,
                                         dlen + 1, size::height, size::width>;

    using rest_size = fwd_problem_size_t<size::batch, size::ifm, size::ofm_sets,
                                         dlen, size::height, size::width>;

    static const long_t other_ioffset =
        Problem::ioffset + shapes::input::depth * (dlen + 1) * drest;

    static const long_t other_ooffset =
        Problem::ooffset + shapes::output::depth * (dlen + 1) * drest;

public:
    template <long_t K>
    struct part
    {
        using type = typename std::conditional<
            (K < drest),
            fwd_problem_t<
                Problem::threads / N, part_size,
                Problem::ioffset + shapes::input::depth*(dlen + 1) * K,
                Problem::ooffset + shapes::output::depth*(dlen + 1) * K,
                Problem::koffset, Problem::boffset, shapes>,
            fwd_problem_t<
                Problem::threads / N, rest_size,
                other_ioffset + shapes::input::depth * dlen*(K - drest),
                other_ooffset + shapes::output::depth * dlen*(K - drest),
                Problem::koffset, Problem::boffset, shapes>>::type;
    };

    using rest = null_fwd_problem_t;
};

template <class Problem, long_t N>
struct fwd_height_splitn_t
{
private:
    using size   = typename Problem::size;
    using shapes = typename Problem::shapes;

    static const long_t hlen  = size::height / N;
    static const long_t hrest = size::height % N;

    using part_size = fwd_problem_size_t<size::batch, size::ifm, size::ofm_sets,
                                         size::depth, hlen + 1, size::width>;

    using rest_size = fwd_problem_size_t<size::batch, size::ifm, size::ofm_sets,
                                         size::depth, hlen, size::width>;

    static const long_t other_ioffset =
        Problem::ioffset + shapes::input::height * (hlen + 1) * hrest;

    static const long_t other_ooffset =
        Problem::ooffset + shapes::output::height * (hlen + 1) * hrest;

public:
    template <long_t K>
    struct part
    {
        using type = typename std::conditional<
            (K < hrest),
            fwd_problem_t<
                Problem::threads / N, part_size,
                Problem::ioffset + shapes::input::height*(hlen + 1) * K,
                Problem::ooffset + shapes::output::height*(hlen + 1) * K,
                Problem::koffset, Problem::boffset, shapes>,
            fwd_problem_t<
                Problem::threads / N, rest_size,
                other_ioffset + shapes::input::height * hlen*(K - hrest),
                other_ooffset + shapes::output::height * hlen*(K - hrest),
                Problem::koffset, Problem::boffset, shapes>>::type;
    };

    using rest = null_fwd_problem_t;
};

template <class Problem, long_t N>
struct fwd_width_splitn_t
{
private:
    using size   = typename Problem::size;
    using shapes = typename Problem::shapes;

    static const long_t wlen  = size::width / N;
    static const long_t wrest = size::width % N;

    using part_size = fwd_problem_size_t<size::batch, size::ifm, size::ofm_sets,
                                         size::depth, size::height, wlen + 1>;

    using rest_size = fwd_problem_size_t<size::batch, size::ifm, size::ofm_sets,
                                         size::depth, size::height, wlen>;

    static const long_t other_ioffset =
        Problem::ioffset + shapes::input::width * (wlen + 1) * wrest;

    static const long_t other_ooffset =
        Problem::ooffset + shapes::output::width * (wlen + 1) * wrest;

public:
    template <long_t K>
    struct part
    {
        using type = typename std::conditional<
            (K < wrest),
            fwd_problem_t<
                Problem::threads / N, part_size,
                Problem::ioffset + shapes::input::width*(wlen + 1) * K,
                Problem::ooffset + shapes::output::width*(wlen + 1) * K,
                Problem::koffset, Problem::boffset, shapes>,
            fwd_problem_t<
                Problem::threads / N, rest_size,
                other_ioffset + shapes::input::width * wlen*(K - wrest),
                other_ooffset + shapes::output::width * wlen*(K - wrest),
                Problem::koffset, Problem::boffset, shapes>>::type;
    };

    using rest = null_fwd_problem_t;
};

template <class Problem, long_t N>
struct fwd_nosplitn_t
{
    template <long_t K>
    struct part
    {
        using type = typename std::conditional<
            (K > 0), null_fwd_problem_t,
            fwd_problem_t<1, typename Problem::size, Problem::ioffset,
                          Problem::ooffset, Problem::koffset, Problem::boffset,
                          typename Problem::shapes>>::type;
    };

    using rest = null_fwd_problem_t;
};

template <class Problem, long_t N, long_t Ratio>
struct fwd_splitn_problem_t
{
    using type = typename conditional_t<
        condition_t<(Ratio >= 2048)>, fwd_nosplitn_t<Problem, N>,
#if (SIMD_WIDTH > 8)
// condition_t< ((Problem::size::depth/N)>=64)&&(Problem::size::depth%N==0) >,
// fwd_depth_splitn_t< Problem, N >,
// condition_t< ((Problem::size::height/N)>=128) >,
// fwd_height_splitn_t< Problem, N >,
// condition_t< ((Problem::size::height/N)>=64)&&(Problem::size::height%N==0) >,
// fwd_height_splitn_t< Problem, N >,
// condition_t< ((Problem::size::width/N)>=32)&&(Problem::size::width%N==0) >,
// fwd_width_splitn_t< Problem, N >,
#else
// condition_t< (Problem::size::height/N)>=50 >,
// fwd_height_splitn_t< Problem, N >,
// condition_t< (Problem::size::width/N)>=50 >,
// fwd_height_splitn_t< Problem, N >,
// condition_t< (Problem::threads>32)&&(Problem::size::height >
// Problem::size::width) >,
// fwd_height_splitn_t< Problem, N >,
// condition_t< (Problem::threads>32) >,
// fwd_width_splitn_t< Problem, N >,

#endif
        condition_t<(Problem::size::batch >= N * 2)>,
        fwd_batch_splitn_t<Problem, N>,
        condition_t<(Problem::size::ofm_sets >= N * 2)>,
        fwd_ofm_splitn_t<Problem, N>,
        condition_t<(Problem::size::depth >= N * 2)>,
        fwd_depth_splitn_t<Problem, N>,
        condition_t<(Problem::size::height >= N * 2)>,
        fwd_height_splitn_t<Problem, N>,
        condition_t<(Problem::size::width >= N) && (Problem::size::width >= 7)>,
        fwd_width_splitn_t<Problem, N>, fwd_nosplitn_t<Problem, N>>::type;
};
}
} // namespace znn:phi
