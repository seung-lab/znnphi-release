#pragma once

#include "znn/types.hpp"

#include <functional>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>

#if !defined(ZNN_NUM_CORES)
#define ZNN_NUM_CORES 32
#endif

namespace znn { namespace phi {

class kernel_launcher
{
private:
    std::mutex              m          ;
    std::condition_variable master_cv  ;
    std::condition_variable slave_cv   ;

    long_t const num_cpus              ;
    long_t const threads_per_cpu       ;
    long_t const num_threads           ;
    long_t idle_threads                ;
    long_t epoch                       ;

    std::function<void()>*  kernels    ;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_times[1024];

public:
    long_t available_threads() const
    {
        return num_threads;
    }

    long_t available_cpus() const
    {
        return num_cpus;
    }

    long_t available_threads_per_cpu() const
    {
        return threads_per_cpu;
    }

private:
    void thread_loop( long_t id, long_t core )
    {
        cpu_set_t old_set;
        sched_getaffinity(0, sizeof(old_set), &old_set);

        cpu_set_t set;
        CPU_ZERO( &set );

        CPU_SET( static_cast<int>(core), &set );
        sched_setaffinity(0, sizeof(set), &set);

        long_t next_epoch = 1;
        //std::function<void()>* local_kernels;

        while ( 1 )
        {

            {
                std::unique_lock<std::mutex> g(m);
                if ( ++idle_threads == num_threads )
                {
                    master_cv.notify_all();
                }
                while ( next_epoch > epoch )
                {
                    slave_cv.wait(g);
                }
            }

            ++next_epoch;

            if ( kernels == nullptr )
            {
                std::unique_lock<std::mutex> g(m);
                if ( ++idle_threads == num_threads )
                {
                    master_cv.notify_all();
                }
                sched_setaffinity(0, sizeof(old_set), &old_set);
                return;
            }
            else if ( kernels[id] )
            {
                kernels[id]();
                end_times[id] = std::chrono::high_resolution_clock::now();
            }
        }
    }

public:
    kernel_launcher( long_t n_cpus, long_t n_hwt, long_t cpu_offset = 0 )
        : m()
        , master_cv()
        , slave_cv()
        , num_cpus(n_cpus)
        , threads_per_cpu(n_hwt)
        , num_threads(n_cpus*n_hwt)
        , idle_threads(0)
        , epoch(0)
        , kernels(nullptr)
    {
        std::unique_lock<std::mutex> g(m);

        for ( long_t c = 0; c < n_cpus; ++c )
        {
            for ( long_t h = 0; h < n_hwt; ++h )
            {
                long_t id   = c * n_hwt + h;
                long_t core = c + cpu_offset + h * ZNN_NUM_CORES;

                std::thread t( &kernel_launcher::thread_loop, this, id, core );
                t.detach();
            }
        }
        while ( idle_threads < num_threads )
        {
            master_cv.wait(g);
        }
    }

    void launch( std::function<void()>* ks )
    {
        std::unique_lock<std::mutex> g(m);

        start_time = std::chrono::high_resolution_clock::now();

        kernels = ks;
        ++epoch;
        idle_threads = 0;
        slave_cv.notify_all();

        while ( idle_threads < num_threads )
        {
            master_cv.wait(g);
        }

        end_time = std::chrono::high_resolution_clock::now();
    }

    void report()
    {
        for ( long_t i = 0; i < num_threads; ++i )
        {
            auto duration1
                = std::chrono::duration_cast<std::chrono::microseconds>
                (end_times[i]-start_time).count();

            auto duration2
                = std::chrono::duration_cast<std::chrono::microseconds>
                (end_time-end_times[i]).count();

            std::cout << "Thread : " << i << " took " << duration1
                      << " and finised this much before the last one "
                      << duration2 << "\n";
            std::cout << "\nTOTAL TIME: "
                      << std::chrono::duration_cast<std::chrono::microseconds>
                (end_time-start_time).count() << "\n";
        }
    }

    ~kernel_launcher()
    {
        launch(nullptr);
    }

};

}} // namespace znn::phi
