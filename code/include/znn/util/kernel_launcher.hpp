#pragma once

#include "znn/types.hpp"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <pthread.h>
#include <thread>

#if !defined(ZNN_NUM_CORES)
#define ZNN_NUM_CORES 32
#endif

namespace znn
{
namespace phi
{

class kernel_launcher
{
private:
    pthread_barrier_t      barrier;
    std::function<void()>* kernels;
    cpu_set_t              old_set_;
    long_t                 num_threads_;

private:
    void thread_loop(long_t id, long_t core)
    {
        cpu_set_t old_set;
        sched_getaffinity(0, sizeof(old_set), &old_set);

        cpu_set_t set;
        CPU_ZERO(&set);

        CPU_SET(static_cast<int>(core), &set);
        sched_setaffinity(0, sizeof(set), &set);

        pthread_barrier_wait(&barrier);

        // Constructor done

        while (1)
        {
            pthread_barrier_wait(&barrier);

            if (kernels == nullptr)
            {
                sched_setaffinity(0, sizeof(old_set), &old_set);
                pthread_barrier_wait(&barrier);
                return;
            }
            else if (kernels[id])
            {
                kernels[id]();
            }

            pthread_barrier_wait(&barrier);
        }
    }

public:
    kernel_launcher(long_t n_cpus, long_t n_hwt, long_t cpu_offset = 0)
        : kernels(nullptr)
        , num_threads_(n_cpus * n_hwt)
    {
        sched_getaffinity(0, sizeof(old_set_), &old_set_);

        cpu_set_t set;
        CPU_ZERO(&set);

        CPU_SET(static_cast<int>(0), &set);
        sched_setaffinity(0, sizeof(set), &set);

        pthread_barrier_init(&barrier, NULL, static_cast<int>(n_cpus * n_hwt));

        for (long_t c = 0; c < n_cpus; ++c)
        {
            for (long_t h = 0; h < n_hwt; ++h)
            {
                if (c + h > 0)
                {
                    long_t id   = c * n_hwt + h;
                    long_t core = c + cpu_offset + h * ZNN_NUM_CORES;

                    std::thread t(&kernel_launcher::thread_loop, this, id,
                                  core);
                    t.detach();
                }
            }
        }

        pthread_barrier_wait(&barrier);
    }

    void launch(std::function<void()>* ks)
    {
        kernels = ks;

        pthread_barrier_wait(&barrier);

        if (kernels == nullptr)
        {
            sched_setaffinity(0, sizeof(old_set_), &old_set_);
        }
        else if (kernels[0])
        {
            kernels[0]();
        }

        pthread_barrier_wait(&barrier);
    }

    long_t num_threads() const { return num_threads_; }

    ~kernel_launcher()
    {
        launch(nullptr);
        pthread_barrier_destroy(&barrier);
    }
};
}
} // namespace znn::phi
