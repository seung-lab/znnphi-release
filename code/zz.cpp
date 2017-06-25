#include "znn/bench/update_full_kernel.hpp"
#include "znn/types.hpp"
#include "znn/util/kernel_launcher.hpp"

using namespace znn::phi;

int main()
{
    print_compiler_version();

    // auto x = propagation::pad_bounds<100, 3, 1>::bounds;

    // for (long_t i = 0; i < 100; ++i)
    // {
    //     std::cout << x[i * 2] << " - " << x[i * 2 + 1] << "\n";
    // }

    benchmark_update_full_kernel<1, 568, 568, 1, 3, 3>(2000);
    benchmark_update_full_kernel<1, 280, 280, 1, 3, 3>(2000);
    benchmark_update_full_kernel<1, 136, 136, 1, 3, 3>(2000);
    benchmark_update_full_kernel<1, 64, 64, 1, 3, 3>(4000);
    benchmark_update_full_kernel<1, 28, 28, 1, 3, 3>(8000);
    // benchmark_propagation_full_image<false, 1, 52, 52, 1, 11, 11, 1, 4,
    // 4>(20000);
    // benchmark_propagation_full_image<true, 1, 112, 112, 1, 3, 3, 1, 1,
    // 1>(2000);
    // benchmark_propagation_sub_layer<1, 1, 1, 568, 568, 1, 3, 3, 1, 1,
    // 1>(2000);

    {
        std::function<void()> fns[4];

        fns[0] = fns[1] = fns[2] = fns[3] = []() {
            benchmark_update_full_kernel<1, 64, 64, 1, 3, 3>(15000);
        };

        {
            kernel_launcher kl(1, 2, 0);
            kl.launch(fns);
        }
    }
    //     {
    //         kernel_launcher kl(1, 4, 0);
    //         kl.launch(fns);
    //     }
    // }

    // benchmark_propagation_full_image_pad<false, 4, 12, 12, 3, 3, 3, 1, 1, 1,
    // 1,
    //                                      1, 1>(20010);

    // {
    //     std::function<void()> fns[4];

    //     fns[0] = fns[1] = fns[2] = fns[3] = []() {
    //         benchmark_propagation_full_image_pad<false, 4, 12, 12, 3, 3, 3,
    //         1,
    //                                              1, 1, 1, 1, 1>(20010);
    //     };

    //     {
    //         kernel_launcher kl(1, 2, 0);
    //         kl.launch(fns);
    //     }
    //     {
    //         kernel_launcher kl(1, 4, 0);
    //         kl.launch(fns);
    //     }
    // }
}
