#pragma once

#include "znn/types.hpp"
#include "znn/intrin.hpp"
#include "znn/util/kernel_launcher.hpp"
#include "znn/layer/task.hpp"
#include "znn/layer/dimension.hpp"
#include "znn/layer/mpf/forward_task.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <iostream>
#include <map>

namespace znn { namespace phi {

template< class I,
          class O,
          class K >
class mpf_forward_plan
{
private:
    // static asserts
    static_assert ( I::b * K::d * K::h * K::w == O::b, "bad batch" );

    static_assert ( I::d % K::d == K::d - 1, "bad d" );
    static_assert ( I::h % K::h == K::h - 1, "bad h" );
    static_assert ( I::w % K::w == K::w - 1, "bad w" );

    static_assert ( I::d / K::d == O::d, "bad o d" );
    static_assert ( I::h / K::h == O::h, "bad o h" );
    static_assert ( I::w / K::w == O::w, "bad o w" );

    static_assert ( I::f == O::f, "bad f" );

    static const long_t SIMD_FM = ( I::f + SIMD_WIDTH - 1 ) / SIMD_WIDTH;

    static const long_t OB_STRIDE = O::b * K::d * K::h * K::w;

    float const * __restrict input ;
    float       * __restrict output;

    std::vector<std::unique_ptr<task>> tasks;

    kernel_launcher * launcher;
    std::vector<std::function<void()>> fns  ;

private:


    void execute_some_tasks( long_t first, long_t last )
    {
        for ( ; first < last; ++first )
        {
            tasks[first]->execute();
        }
    }

    std::vector<std::unique_ptr<task>> create_tasks()
    {
        std::vector<std::unique_ptr<task>> ret;

        for ( long_t d = 0; d < O::d; ++d )
            for ( long_t h = 0; h < O::h; ++h )
                ret.push_back(
                    std::make_unique<
                    mpf_forward_task<
                    dimension<1,I::ds,O::ds>,
                    dimension<1,I::hs,O::hs>,
                    dimension<O::w,I::ws,O::ws>,
                    K::d, K::h, K::w, O::bs>>
                    ( input + ( d * K::d * I::ds + h * K::h * I::hs ) * SIMD_WIDTH,
                      output + ( d * O::ds + h * O::hs ) * SIMD_WIDTH ));
        return ret;
    }

    void partial_schedule( long_t task_from,
                           long_t task_n,
                           long_t thread_from,
                           long_t thread_n )
    {
        if ( thread_n <= 4 )
        {
            long_t per_thread = (task_n + thread_n - 1) / thread_n;
            if ( per_thread == 0 ) per_thread = 1;

            for ( long_t i = 0; i < thread_n; ++i )
            {
                fns[thread_from + i ]
                    = std::bind( &mpf_forward_plan::execute_some_tasks, this,
                                 task_from + std::min( i * per_thread, task_n ),
                                 task_from + std::min( (i+1) * per_thread, task_n ) );
            }
        }
        else if ( (task_n%3==0) && (thread_n%3==0) )
        {
            partial_schedule( task_from, task_n / 3,
                              thread_from, thread_n / 3 );

            partial_schedule( task_from + task_n / 3, task_n / 3,
                              thread_from + thread_n / 3, thread_n / 3 );

            partial_schedule( task_from + 2*task_n / 3, task_n / 3,
                              thread_from + 2*thread_n / 3, thread_n / 3 );
        }
        else if ( (task_n%2==0) && (thread_n%2==0) )
        {
            partial_schedule( task_from, task_n / 2,
                              thread_from, thread_n / 2 );

            partial_schedule( task_from + task_n / 2, task_n / 2,
                              thread_from + thread_n / 2, thread_n / 2 );
        }
        else
        {
            long_t per_thread = (task_n + thread_n - 1) / thread_n;
            if ( per_thread == 0 ) per_thread = 1;

            for ( long_t i = 0; i < thread_n; ++i )
            {
                fns[thread_from + i ]
                    = std::bind( &mpf_forward_plan::execute_some_tasks, this,
                                 task_from + std::min( i * per_thread, task_n ),
                                 task_from + std::min( (i+1) * per_thread, task_n ) );
        }


        }

    }

    void schedule()
    {
        partial_schedule(0,tasks.size(),0,fns.size());
    }

public:

    long_t flops() const
    {
        return O::b * O::f * O::d * O::h * O::w * K::d * K::h * K::w;
    }

    double gflops() const
    {
        return static_cast<double>(flops()) / 1000000000;
    }

    mpf_forward_plan( kernel_launcher * kl,
                      float const * __restrict i,
                      float       * __restrict o )
        : input(i)
        , output(o)
        , tasks()
        , launcher(kl)
        , fns(kl->available_threads())
    {

        tasks = create_tasks();

        std::cout << "Plan created with: " << tasks.size() << " tasks\n";
        std::cout << "          FLOPS  : " << flops() << "\n";
        std::cout << "         GFLOPS  : " << gflops() << "\n";

        long_t multiplier = SIMD_FM * I::b;

        //auto ot = order_tasks();

        std::cout << "MULT:              " << multiplier << "\n";

        std::cout << "Plan created with: " << tasks.size() << " tasks\n";
        std::cout << "Full task size   : " << tasks.size() * multiplier << " tasks\n";

        long_t num_tasks = tasks.size();
        long_t total_num_tasks = num_tasks * multiplier;

        tasks.resize( total_num_tasks );

        long_t next = num_tasks;
        for ( long_t b = 0; b < I::b; ++b )
        {
            for ( long_t ofm = 0; ofm < SIMD_FM; ++ofm )
            {
                if ( b || ofm )
                {
                    for ( long_t i = 0; i < num_tasks; ++i )
                    {
                        tasks[next+i] =
                            tasks[i]->offset_copy( b * I::bs + ofm * I::fs * SIMD_WIDTH,
                                                   b * OB_STRIDE + ofm * O::fs * SIMD_WIDTH, 0 );

                    }
                    next += num_tasks;
                }
            }
        }

        schedule();
    }

    void execute()
    {
        launcher->launch( &(fns[0]) );
    }

};



}} // namespace znn:phi
