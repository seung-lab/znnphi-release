#pragma once

//#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <zi/vl/vl.hpp>
#include <complex>
#include <list>
#include <map>
#include <vector>

// GCC version
#if defined(__GNUC__) && !defined(GCC_VERSION) && !defined(__clang__)
#define GCC_VERSION                                                            \
    ((__GNUC__)*10000 + (__GNUC_MINOR__)*100 + (__GNUC_PATCHLEVEL__))
#endif

// Clang version
#if defined(__clang__)
#define CLANG_VERSION                                                          \
    ((__clang_major__)*10000 + (__clang_minor__)*100 + (__clang_patchlevel__))
#endif

// ICC version
#if defined(__INTEL_COMPILER)
#define ICC_VERSION __INTEL_COMPILER
#endif

inline void print_verson(int v)
{
    std::cout << (v / 10000) << "." << ((v % 10000) / 100) << "." << (v % 100);
}

inline void print_compiler_version()
{
#if defined(__INTEL_COMPILER)
    std::cout << "icc " << __INTEL_COMPILER << " compat with ";
#endif
#if defined(__GNUC__) && !defined(__clang__)
    std::cout << "gcc ";
    print_verson(GCC_VERSION);
#elif defined(__clang__)
    std::cout << "clang ";
    print_verson(CLANG_VERSION);
#endif
    std::cout << std::endl;
}

#define ZNN_PRAGMA(x) _Pragma(#x)

#define znn_inline __attribute__((always_inline)) inline

#define znn_when(cond, type) typename std::enable_if<(cond), type>::type

#define znn_inline_when(cond, type) znn_inline znn_when(cond, type)

#if defined(ZNN_PHI_AVX512) && !defined(ZNN_PHI_AVX2)
#define ZNN_PHI_AVX2
#endif

#if defined(ZNN_PHI_AVX2) && !defined(ZNN_PHI_AVX)
#define ZNN_PHI_AVX
#endif

#if defined(ZNN_PHI_ALIGNMENT)
#undef ZNN_PHI_ALIGNMENT
#endif

#if defined(ZNN_PHI_AVX512)
#define ZNN_PHI_ALIGNMENT 64
#elif defined(ZNN_PHI_AVX)
#define ZNN_PHI_ALIGNMENT 32
#else
#define ZNN_PHI_ALIGNMENT 8
#endif

#if defined(ZNN_STD1Y)

namespace std
{
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}

#endif

namespace znn
{
namespace phi
{

typedef std::complex<float> cplx;
typedef std::complex<float> complex;

typedef int64_t long_t;

typedef zi::vl::vec<long_t, 2> vec2i;
typedef zi::vl::vec<long_t, 3> vec3i;
typedef zi::vl::vec<long_t, 4> vec4i;
typedef zi::vl::vec<long_t, 5> vec5i;

typedef std::size_t size_t;

typedef std::lock_guard<std::mutex> guard;

template <typename T>
struct type_t
{
};

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

namespace vectorize
{

struct default_tag
{
};
struct avx_tag
{
};
struct avx2_tag
{
};
struct avx512_tag
{
};
}

namespace detail
{

struct host_pointer_tag
{
};
struct device_pointer_tag
{
};
struct hbw_pointer_tag
{
};

template <typename T, typename Tag>
class pointer
{
public:
    typedef T  value_type;
    typedef T* pointer_type;

private:
    pointer_type ptr_;

public:
    explicit pointer(pointer_type p = nullptr)
        : ptr_(p)
    {
    }

    pointer(std::nullptr_t)
        : ptr_(nullptr)
    {
    }

    pointer(pointer const& other)
        : ptr_(other.get())
    {
    }

    template <typename O>
    pointer(pointer<O, Tag> const& other,
            typename std::enable_if<std::is_convertible<O, T>::value,
                                    void*>::type = 0)
        : ptr_(other.get())
    {
    }

    pointer& operator=(pointer const& other)
    {
        ptr_ = other.ptr_;
        return *this;
    }

    template <typename O>
    typename std::enable_if<std::is_convertible<O, T>::value, pointer&>::type
    operator=(pointer<O, Tag> const& other)
    {
        ptr_ = other.get();
        return *this;
    }

    pointer_type get() const { return ptr_; }

    operator bool() const { return ptr_ != nullptr; }
};

template <typename T, typename charT, typename traits>
std::basic_ostream<charT, traits>&
operator<<(std::basic_ostream<charT, traits>&  os,
           pointer<T, host_pointer_tag> const& p)
{
    os << "h[" << p.get() << "]";
    return os;
}

template <typename T, typename charT, typename traits>
std::basic_ostream<charT, traits>&
operator<<(std::basic_ostream<charT, traits>& os,
           pointer<T, hbw_pointer_tag> const& p)
{
    os << "x[" << p.get() << "]";
    return os;
}

template <typename T, typename charT, typename traits>
std::basic_ostream<charT, traits>&
operator<<(std::basic_ostream<charT, traits>&    os,
           pointer<T, device_pointer_tag> const& p)
{
    os << "d[" << p.get() << "]";
    return os;
}

} //  Namespace detail

template <typename T>
using host_ptr = detail::pointer<T, detail::host_pointer_tag>;

template <typename T>
using hbw_ptr = detail::pointer<T, detail::hbw_pointer_tag>;

template <typename T>
using device_ptr = detail::pointer<T, detail::device_pointer_tag>;

inline constexpr long_t vector_align(long_t bytes)
{
    return ((bytes + ZNN_PHI_ALIGNMENT - 1) / ZNN_PHI_ALIGNMENT) *
           ZNN_PHI_ALIGNMENT;
}
}
} // namespace znn::phi
