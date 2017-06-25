#pragma once

#include "znn/layer/conv/fwd/problem.hpp"
#include "znn/meta.hpp"

namespace znn { namespace phi {

template< class Problem >
struct fwd_batch_split3_t
{
private:
    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch / 3,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch % 3,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width >;


public:

    //static_assert( Problem::size::batch <= 2, "batch size not <= 2" );

    static_assert(true, "I need a semi-column for indenting in emacs");

    using first = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset + shapes::input::batch  * (size::batch / 3),
        Problem::ooffset + shapes::output::batch * (size::batch / 3),
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using third = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset + shapes::input::batch  * (size::batch / 3)*2,
        Problem::ooffset + shapes::output::batch * (size::batch / 3)*2,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using rest = typename std::conditional<
        ( size::batch%3 )==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::batch  * (size::batch - (size::batch%3)),
            Problem::ooffset + shapes::output::batch * (size::batch - (size::batch%3)),
            Problem::koffset,
            Problem::boffset,
            shapes > >::type;
};

template< class Problem >
struct fwd_ofm_split3_t
{
private:
    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets / 3,
                                          size::depth,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets % 3,
                                          size::depth,
                                          size::height,
                                          size::width >;

public:

    static_assert( Problem::size::batch <= 2   , "batch size not <= 2"    );

    using first = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset,
        Problem::ooffset + shapes::output::fm_set * (size::ofm_sets / 3),
        Problem::koffset + shapes::weight::output * (size::ofm_sets / 3),
        Problem::boffset + size::ofm_sets / 3,
        shapes >;

    using third = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset,
        Problem::ooffset + shapes::output::fm_set * (size::ofm_sets / 3)*2,
        Problem::koffset + shapes::weight::output * (size::ofm_sets / 3)*2,
        Problem::boffset + (size::ofm_sets/3) * 2,
        shapes >;

    using rest = typename std::conditional<
        (size::ofm_sets%3)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset,
            Problem::ooffset + shapes::output::fm_set * ( size::ofm_sets - (size::ofm_sets%3) ),
            Problem::koffset + shapes::weight::output * ( size::ofm_sets - (size::ofm_sets%3) ),
            Problem::boffset + size::ofm_sets - (size::ofm_sets%3),
            shapes > >::type;
};


template< class Problem >
struct fwd_depth_split3_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth / 3,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth % 3,
                                          size::height,
                                          size::width >;

public:

    static_assert( Problem::size::batch <= 2   , "batch size not <= 2"    );
    static_assert( Problem::size::ofm_sets <= 2, "ofm_sets size not <= 2" );

    using first = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset + shapes::input::depth  * (size::depth / 3),
        Problem::ooffset + shapes::output::depth * (size::depth / 3),
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using third = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset + shapes::input::depth  * (size::depth / 3)*2,
        Problem::ooffset + shapes::output::depth * (size::depth / 3)*2,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using rest = typename std::conditional<
        (size::depth%3)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::depth  * (size::depth-(size::depth%3)),
            Problem::ooffset + shapes::output::depth * (size::depth-(size::depth%3)),
            Problem::koffset,
            Problem::boffset,
            shapes > >::type;
};

template< class Problem >
struct fwd_height_split3_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height / 3,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height % 3,
                                          size::width >;

public:

    static_assert( Problem::size::batch <= 2   , "batch size not <= 2"    );
    static_assert( Problem::size::ofm_sets <= 2, "ofm_sets size not <= 2" );
    static_assert( Problem::size::depth <= 2   , "depth size not <= 2"    );

    using first = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset + shapes::input::height  * (size::height / 3),
        Problem::ooffset + shapes::output::height * (size::height / 3),
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using third = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset + shapes::input::height  * (size::height / 3)*2,
        Problem::ooffset + shapes::output::height * (size::height / 3)*2,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using rest = typename std::conditional<
        (size::height%3)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::height  * (size::height-(size::height%3)),
            Problem::ooffset + shapes::output::height * (size::height-(size::height%3)),
            Problem::koffset,
            Problem::boffset,
            shapes > >::type;
};

template< class Problem >
struct fwd_width_split3_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width / 3 >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width % 3 >;

public:

    static_assert( Problem::size::batch <= 2   , "batch size not <= 2"    );
    static_assert( Problem::size::ofm_sets <= 2, "ofm_sets size not <= 2" );
    static_assert( Problem::size::depth <= 2   , "depth size not <= 2"    );
    static_assert( Problem::size::height <= 2  , "height size not <= 2"   );

    using first = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using second = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset + shapes::input::width  * (size::width / 3),
        Problem::ooffset + shapes::output::width * (size::width / 3),
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using third = fwd_problem_t<
        Problem::threads / 3,
        part_size,
        Problem::ioffset + shapes::input::width  * (size::width / 3)*2,
        Problem::ooffset + shapes::output::width * (size::width / 3)*2,
        Problem::koffset,
        Problem::boffset,
        shapes >;

    using rest = typename std::conditional<
        (size::width%3)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::width  * (size::width-(size::width%3)),
            Problem::ooffset + shapes::output::width * (size::width-(size::width%3)),
            Problem::koffset,
            Problem::boffset,
            shapes > >::type;
};

template< class Problem >
struct fwd_nosplit3_t
{
    using first = fwd_problem_t<
        1,
        typename Problem::size,
        Problem::ioffset,
        Problem::ooffset,
        Problem::koffset,
        Problem::boffset,
        typename Problem::shapes >;

    using second = null_fwd_problem_t;
    using third  = null_fwd_problem_t;
    using rest   = null_fwd_problem_t;
};

template< class Problem >
struct fwd_split3_problem_t
{
    using type = typename conditional_t<
        condition_t< (Problem::size::batch>2) >,
        fwd_batch_split3_t< Problem >,
        condition_t< (Problem::size::ofm_sets>2) >,
        fwd_ofm_split3_t< Problem >,
        condition_t< ((Problem::size::depth%3)==0) >,
        fwd_depth_split3_t< Problem >,
        condition_t< ((Problem::size::height%3)==0) >,
        fwd_height_split3_t< Problem >,
        condition_t< ((Problem::size::width%3)==0) >,
        fwd_width_split3_t< Problem >,
        condition_t< (Problem::size::depth>2) >,
        fwd_depth_split3_t< Problem >,
        condition_t< (Problem::size::height>2) >,
        fwd_height_split3_t< Problem >,
        condition_t< (Problem::size::width>10) >,
        fwd_width_split3_t< Problem >,
        fwd_nosplit3_t< Problem > >::type;
};



}} // namespace znn:phi
