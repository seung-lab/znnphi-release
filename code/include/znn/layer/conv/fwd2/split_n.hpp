#pragma once

#include "znn/layer/conv/fwd2/problem.hpp"
#include "znn/meta.hpp"

namespace znn { namespace phi {

template< class Problem, long_t N >
struct fwd_batch_splitn_t
{
private:
    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch / N,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch % N,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width >;


public:

    //static_assert( Problem::size::batch <= 2, "batch size not <= 2" );

    static_assert(true, "I need a semi-column for indenting in emacs");

    template< long_t K >
    struct part
    {
        using type  = fwd_problem_t<
            Problem::threads / N,
            part_size,
            Problem::ioffset + shapes::input::batch  * (size::batch / N)*K,
            Problem::ooffset + shapes::output::batch * (size::batch / N)*K,
            Problem::koffset,
            Problem::boffset,
            shapes >;
    };

    using rest = typename std::conditional<
        ( size::batch%N )==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::batch  * (size::batch - (size::batch%N)),
            Problem::ooffset + shapes::output::batch * (size::batch - (size::batch%N)),
            Problem::koffset,
            Problem::boffset,
            shapes > >::type;
};

template< class Problem, long_t N >
struct fwd_ofm_splitn_t
{
private:
    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets / N,
                                          size::depth,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets % N,
                                          size::depth,
                                          size::height,
                                          size::width >;

public:

    //static_assert( Problem::size::batch <= N   , "batch size not <= N"    );

    template< long_t K >
    struct part
    {

        using type = fwd_problem_t<
            Problem::threads / N,
            part_size,
            Problem::ioffset,
            Problem::ooffset + shapes::output::fm_set * (size::ofm_sets / N)*K,
            Problem::koffset + shapes::weight::output * (size::ofm_sets / N)*K,
            Problem::boffset + (size::ofm_sets/N)*K,
            shapes >;
    };

    using rest = typename std::conditional<
        (size::ofm_sets%N)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset,
            Problem::ooffset + shapes::output::fm_set * ( size::ofm_sets - (size::ofm_sets%N) ),
            Problem::koffset + shapes::weight::output * ( size::ofm_sets - (size::ofm_sets%N) ),
            Problem::boffset + size::ofm_sets - (size::ofm_sets%N),
            shapes > >::type;
};


template< class Problem, long_t N >
struct fwd_depth_splitn_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth / N,
                                          size::height,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth % N,
                                          size::height,
                                          size::width >;

public:

    //static_assert( Problem::size::batch <= N   , "batch size not <= N"    );
    //static_assert( Problem::size::ofm_sets <= N, "ofm_sets size not <= N" );

    template< long_t K >
    struct part
    {
        using type = fwd_problem_t<
            Problem::threads / N,
            part_size,
            Problem::ioffset + shapes::input::depth  * (size::depth / N)*K,
            Problem::ooffset + shapes::output::depth * (size::depth / N)*K,
            Problem::koffset,
            Problem::boffset,
            shapes >;
    };

    using rest = typename std::conditional<
        (size::depth%N)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::depth  * (size::depth-(size::depth%N)),
            Problem::ooffset + shapes::output::depth * (size::depth-(size::depth%N)),
            Problem::koffset,
            Problem::boffset,
            shapes > >::type;
};

template< class Problem, long_t N >
struct fwd_height_splitn_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height / N,
                                          size::width >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height % N,
                                          size::width >;

public:

    //static_assert( Problem::size::batch <= N   , "batch size not <= N"    );
    //static_assert( Problem::size::ofm_sets <= N, "ofm_sets size not <= N" );
    //static_assert( Problem::size::depth <= N   , "depth size not <= N"    );

    template< long_t K >
    struct part
    {
        using type = fwd_problem_t<
            Problem::threads / N,
            part_size,
            Problem::ioffset + shapes::input::height  * (size::height / N)*K,
            Problem::ooffset + shapes::output::height * (size::height / N)*K,
            Problem::koffset,
            Problem::boffset,
            shapes >;
    };

    using rest = typename std::conditional<
        (size::height%N)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::height  * (size::height-(size::height%N)),
            Problem::ooffset + shapes::output::height * (size::height-(size::height%N)),
            Problem::koffset,
            Problem::boffset,
            shapes > >::type;
};

template< class Problem, long_t N >
struct fwd_width_splitn_t
{
private:

    using size   = typename Problem::size  ;
    using shapes = typename Problem::shapes;

    using part_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width / N >;

    using rest_size = fwd_problem_size_t< size::batch,
                                          size::ifm,
                                          size::ofm_sets,
                                          size::depth,
                                          size::height,
                                          size::width % N >;

public:

    // static_assert( Problem::size::batch <= N   , "batch size not <= N"    );
    // static_assert( Problem::size::ofm_sets <= N, "ofm_sets size not <= N" );
    // static_assert( Problem::size::depth <= N   , "depth size not <= N"    );
    // static_assert( Problem::size::height <= N  , "height size not <= N"   );

    template< long_t K >
    struct part
    {
        using type = fwd_problem_t<
            Problem::threads / N,
            part_size,
            Problem::ioffset + shapes::input::width  * (size::width / N)*K,
            Problem::ooffset + shapes::output::width * (size::width / N)*K,
            Problem::koffset,
            Problem::boffset,
            shapes >;
    };

    using rest = typename std::conditional<
        (size::width%N)==0,
        null_fwd_problem_t,
        fwd_problem_t<
            Problem::threads,
            rest_size,
            Problem::ioffset + shapes::input::width  * (size::width-(size::width%N)),
            Problem::ooffset + shapes::output::width * (size::width-(size::width%N)),
            Problem::koffset,
            Problem::boffset,
            shapes > >::type;
};

template< class Problem, long_t N >
struct fwd_nosplitn_t
{
    template< long_t K >
    struct part
    {
        using type = typename std::conditional<
            (K>0),
            null_fwd_problem_t,
            fwd_problem_t<
                1,
                typename Problem::size,
                Problem::ioffset,
                Problem::ooffset,
                Problem::koffset,
                Problem::boffset,
                typename Problem::shapes >
                >::type;
    };

    using rest   = null_fwd_problem_t;
};

template< class Problem, long_t N, long_t Ratio >
struct fwd_splitn_problem_t
{
    using type = typename conditional_t<
        condition_t< (Ratio>=2048) >,
        fwd_nosplitn_t< Problem, N >,
#if ( SIMD_WIDTH > 8 )
        condition_t< ((Problem::size::depth/N)>=32)&&(Problem::size::depth%N==0) >,
        fwd_depth_splitn_t< Problem, N >,
        condition_t< ((Problem::size::height/N)>=32)&&(Problem::size::height%N==0) >,
        fwd_height_splitn_t< Problem, N >,
        condition_t< ((Problem::size::width/N)>=32)&&(Problem::size::width%N==0) >,
        fwd_width_splitn_t< Problem, N >,
#endif
        // condition_t< ((Problem::size::depth/N)>=32)&&(Problem::size::depth%N==0) >,
        // fwd_depth_splitn_t< Problem, N >,
        // condition_t< ((Problem::size::height/N)>=32)&&(Problem::size::height%N==0) >,
        // fwd_height_splitn_t< Problem, N >,
        // condition_t< ((Problem::size::width/N)>=32)&&(Problem::size::width%N==0) >,
        // fwd_width_splitn_t< Problem, N >,
        condition_t< (Problem::size::batch>=N) >,
        fwd_batch_splitn_t< Problem, N >,
        condition_t< (Problem::size::ofm_sets>=N) >,
        fwd_ofm_splitn_t< Problem, N >,
        condition_t< (Problem::size::depth>=N) >,
        fwd_depth_splitn_t< Problem, N >,
        condition_t< (Problem::size::height>=N*8) >,
        fwd_height_splitn_t< Problem, N >,
        condition_t< (Problem::size::width>=N) && (Problem::size::width>=7)>,
        fwd_width_splitn_t< Problem, N >,
        fwd_nosplitn_t< Problem, N > >::type;
};



}} // namespace znn:phi
