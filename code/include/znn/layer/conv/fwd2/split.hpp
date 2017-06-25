#pragma once

#include "znn/layer/conv/fwd2/problem.hpp"
#include "znn/meta.hpp"

namespace znn { namespace phi {

template< class Problem >
struct fwd_batch_split_t
{
private:
    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using half_size = fwd_problem_size_t< size::batch / 2,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch % 2,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width >;


public:

    //static_assert( Problem::size::batch == 1, "batch size not 1" );

    static_assert(true, "I need a semi-column for indenting in emacs");

    using first = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset + shapes::input::batch  * size::batch / 2,
        Problem::ooffset + shapes::output::batch * size::batch / 2,
        Problem::koffset,
        shapes >;

    using rest = typename std::conditional<
        ( size::batch%2 )==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::batch  * (size::batch - 1),
            Problem::ooffset + shapes::output::batch * (size::batch - 1),
            Problem::koffset,
            shapes > >::type;
};

template< class Problem >
struct fwd_ofm_split_t
{
private:
    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using half_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets / 2,
                                          size::depth,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets % 2,
                                          size::depth,
                                          size::height,
                                          size::width >;

public:

    static_assert( Problem::size::batch == 1   , "batch size not 1"    );

    using first = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset,
        Problem::ooffset + shapes::output::fm_set * size::ofm_sets / 2,
        Problem::koffset + shapes::weight::output * size::ofm_sets / 2,
        shapes >;

    using rest = typename std::conditional<
        (size::ofm_sets%2)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset,
            Problem::ooffset + shapes::output::fm_set * ( size::ofm_sets - 1 ),
            Problem::koffset + shapes::weight::output * ( size::ofm_sets - 1 ),
            shapes > >::type;
};


template< class Problem >
struct fwd_depth_split_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using half_size = fwd_problem_size_t< 1,
                                          size::ifm,
                                          1,
                                          size::depth / 2,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< 1,
                                          size::ifm,
                                          1,
                                          size::depth % 2,
                                          size::height,
                                          size::width >;

public:

    static_assert( Problem::size::batch == 1   , "batch size not 1"    );
    static_assert( Problem::size::ofm_sets == 1, "ofm_sets size not 1" );

    using first = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset + shapes::input::depth  * size::depth / 2,
        Problem::ooffset + shapes::output::depth * size::depth / 2,
        Problem::koffset,
        shapes >;

    using rest = typename std::conditional<
        (size::depth%2)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::depth  * (size::depth-1),
            Problem::ooffset + shapes::output::depth * (size::depth-1),
            Problem::koffset,
            shapes > >::type;
};

template< class Problem >
struct fwd_height_split_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using half_size = fwd_problem_size_t< 1,
                                          size::ifm,
                                          1,
                                          1,
                                          size::height / 2,
                                          size::width >;

    using rest_size = fwd_problem_size_t< 1,
                                          size::ifm,
                                          1,
                                          1,
                                          size::height % 2,
                                          size::width >;

public:

    static_assert( Problem::size::batch == 1   , "batch size not 1"    );
    static_assert( Problem::size::ofm_sets == 1, "ofm_sets size not 1" );
    static_assert( Problem::size::depth == 1   , "depth size not 1"    );

    using first = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset + shapes::input::height  * size::height / 2,
        Problem::ooffset + shapes::output::height * size::height / 2,
        Problem::koffset,
        shapes >;

    using rest = typename std::conditional<
        (size::height%2)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::height  * (size::height-1),
            Problem::ooffset + shapes::output::height * (size::height-1),
            Problem::koffset,
            shapes > >::type;
};

template< class Problem >
struct fwd_width_split_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using half_size = fwd_problem_size_t< 1,
                                          size::ifm,
                                          1,
                                          1,
                                          1,
                                          size::width / 2 >;

    using rest_size = fwd_problem_size_t< 1,
                                          size::ifm,
                                          1,
                                          1,
                                          1,
                                          size::width % 2 >;

public:

    static_assert( Problem::size::batch == 1   , "batch size not 1"    );
    static_assert( Problem::size::ofm_sets == 1, "ofm_sets size not 1" );
    static_assert( Problem::size::depth == 1   , "depth size not 1"    );
    static_assert( Problem::size::height == 1  , "height size not 1"   );

    using first = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 2,
        half_size,
        Problem::ioffset + shapes::input::width  * size::width / 2,
        Problem::ooffset + shapes::output::width * size::width / 2,
        Problem::koffset,
        shapes >;

    using rest = typename std::conditional<
        (size::width%2)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::width  * (size::width-1),
            Problem::ooffset + shapes::output::width * (size::width-1),
            Problem::koffset,
            shapes > >::type;
};

template< class Problem >
struct fwd_nosplit_t
{
    using first = fwd_problem_t<
        1,
        typename Problem::size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        typename Problem::shapes >;

    using second = null_fwd_problem_t;
    using rest   = null_fwd_problem_t;
};

template< class Problem >
struct fwd_split_problem_t
{
    using type = typename conditional_t<
        condition_t< (Problem::size::batch>1) >,
        fwd_batch_split_t< Problem >,
        condition_t< (Problem::size::ofm_sets>1) >,
        fwd_ofm_split_t< Problem >,
        condition_t< (Problem::size::depth>1) >,
        fwd_depth_split_t< Problem >,
        condition_t< (Problem::size::height>1) >,
        fwd_height_split_t< Problem >,
        condition_t< (Problem::size::width>1) >,
        fwd_width_split_t< Problem >,
        fwd_nosplit_t< Problem > >::type;
};



}} // namespace znn:phi
