#pragma once

#include "znn/tensor/tensor_ref.hpp"

namespace znn
{
namespace phi
{

template <class T, size_t D, class A>
class tensor : public tensor_ref<T, D, A>
{
private:
    typedef tensor_ref<T, D, A> super_type;

public:
    static const size_t dimensionality = super_type::dimensionality;

    typedef typename super_type::element         element;
    typedef typename super_type::value_type      value_type;
    typedef typename super_type::reference       reference;
    typedef typename super_type::const_reference const_reference;
    typedef typename super_type::architecture    architecture;

private:
    void allocate_memory()
    {
        if (super_type::strides_[0] > 0)
        {
            auto ptr = detail::tensor::malloc(this->num_elements() * sizeof(T),
                                              architecture());
            this->ptr_ = reinterpret_cast<T*>(ptr);
        }
    }

private:
    void do_random_init(T const& v, detail::tensor::host_tag)
    {
        random_initialize(this->data(), this->num_elements(), v);
    }

    void do_random_init(T const& v, detail::tensor::hbw_tag)
    {
        random_initialize(this->data(), this->num_elements(), v);
    }

    void do_random_init(T const& v, detail::tensor::device_tag)
    {
        T* rnd = reinterpret_cast<T*>(detail::tensor::malloc(
            this->num_elements() * sizeof(T), detail::tensor::host_tag()));
        random_initialize(rnd, this->num_elements(), v);
        this->load(rnd, detail::tensor::host_tag());
        detail::tensor::free(rnd, detail::tensor::host_tag());
    }

    void do_const_init(detail::tensor::host_tag, T const& v)
    {
        std::fill_n(this->data(), this->num_elements(), static_cast<T>(v));
    }

    void do_const_init(detail::tensor::hbw_tag, T const& v)
    {
        std::fill_n(this->data(), this->num_elements(), static_cast<T>(v));
    }

    void do_const_init(detail::tensor::device_tag, T const& v)
    {
        T* rnd = reinterpret_cast<T*>(detail::tensor::malloc(
            this->num_elements() * sizeof(T), detail::tensor::host_tag()));
        std::fill_n(rnd, this->num_elements(), static_cast<T>(v));
        this->load(rnd, detail::tensor::host_tag());
        detail::tensor::free(rnd, detail::tensor::host_tag());
    }

public:
    void randomize(T const& v = static_cast<T>(0.1))
    {
        do_random_init(v, architecture());
    }

    void set_to_const(T const & v) { do_const_init(architecture(),v); }

    void reset()
    {
        if (this->ptr_)
        {
            detail::tensor::free(this->ptr_, architecture());
            this->ptr_        = nullptr;
            this->strides_[0] = 0;
        }
    }

    ~tensor() { reset(); }

    tensor() noexcept : super_type() {}

    tensor(tensor&& other) noexcept { *this = std::move(other); }

    explicit tensor(zi::vl::vec<long_t, D> const& e)
        : super_type(nullptr, e)
    {
        allocate_memory();
    }

    template <typename... Args>
    explicit tensor(Args&&... args)
        : super_type(nullptr,
                     zi::vl::vec<long_t, D>(std::forward<Args>(args)...))
    {
        allocate_memory();
    }

    explicit tensor(detail::tensor::random_initialize_tag,
                    zi::vl::vec<long_t, D> const& e)
        : super_type(nullptr, e)
    {
        allocate_memory();
        randomize();
    }

    explicit tensor(detail::tensor::one_init_tag,
                    zi::vl::vec<long_t, D> const& e)
        : super_type(nullptr, e)
    {
        allocate_memory();
        set_to_const(1);
    }

    explicit tensor(detail::tensor::zero_init_tag,
                    zi::vl::vec<long_t, D> const& e)
        : super_type(nullptr, e)
    {
        allocate_memory();
        set_to_const(0);
    }

    template <typename... Args>
    explicit tensor(detail::tensor::random_initialize_tag, Args&&... args)
        : super_type(nullptr,
                     zi::vl::vec<long_t, D>(std::forward<Args>(args)...))
    {
        allocate_memory();
        randomize();
    }

    template <typename... Args>
    explicit tensor(detail::tensor::one_init_tag, Args&&... args)
        : super_type(nullptr,
                     zi::vl::vec<long_t, D>(std::forward<Args>(args)...))
    {
        allocate_memory();
        set_to_const(1);
    }

    template <typename... Args>
    explicit tensor(detail::tensor::zero_init_tag, Args&&... args)
        : super_type(nullptr,
                     zi::vl::vec<long_t, D>(std::forward<Args>(args)...))
    {
        allocate_memory();
        set_to_const(0);
    }

    tensor& operator=(tensor&& other) noexcept
    {
        this->ptr_        = other.ptr_;
        other.ptr_        = nullptr;
        this->extents_    = other.extents_;
        this->strides_    = other.strides_;
        other.strides_[0] = 0;
        return *this;
    }

    tensor& operator=(tensor const& other) noexcept
    {
        super_type::operator=(other);
        return *this;
    }

    template <typename Tensor>
    tensor& operator=(Tensor const& other) noexcept
    {
        super_type::operator=(other);
        return *this;
    }
};

template <class T, size_t D>
using host_tensor = tensor<T, D, detail::tensor::host_tag>;

template <class T, size_t D>
using hbw_tensor = tensor<T, D, detail::tensor::hbw_tag>;

template <class T, size_t D>
using device_tensor = tensor<T, D, detail::tensor::device_tag>;

template <class T>
using host_array = tensor<T, 1, detail::tensor::host_tag>;

template <class T>
using hbw_array = tensor<T, 1, detail::tensor::hbw_tag>;

template <class T>
using device_array = tensor<T, 1, detail::tensor::device_tag>;

namespace
{
detail::tensor::host_tag   from_host;
detail::tensor::hbw_tag    from_hbw;
detail::tensor::device_tag from_device;

detail::tensor::host_tag   to_host;
detail::tensor::hbw_tag    to_hbw;
detail::tensor::device_tag to_device;

detail::tensor::random_initialize_tag rand_init;
detail::tensor::one_init_tag          one_init;
detail::tensor::zero_init_tag         zero_init;
}

inline void ____use_tag_instances()
{
    static_cast<void>(from_host);
    static_cast<void>(to_host);
    static_cast<void>(from_hbw);
    static_cast<void>(to_hbw);
    static_cast<void>(from_device);
    static_cast<void>(to_device);
    static_cast<void>(rand_init);
    static_cast<void>(one_init);
    static_cast<void>(zero_init);
}
}
} // namespace znn::phi
