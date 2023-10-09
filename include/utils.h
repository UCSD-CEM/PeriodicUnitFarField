#include <complex>
#include "thrust/complex.h"
#include "thrust/transform.h"
#include "thrust/functional.h"
#include <cassert>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if(err != CUFFT_SUCCESS) { \
        fprintf(stderr, "CUFFT error in %s at line %d: %s\n", \
        __FILE__, __LINE__, cufftGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_DFTI(call) { \
    MKL_LONG err = call; \
    if(err && !DftiErrorClass(err, DFTI_NO_ERROR)) { \
        fprintf(stderr, "DFTI error in %s at line %d: %s\n", \
        __FILE__, __LINE__, DftiErrorMessage(err)); \
        exit(EXIT_FAILURE); \
    } \
}


namespace puff{

    template <typename ValueType, typename MemorySpace>
    void Vector_element_wise_multiply_Vector(const Vector<ValueType, MemorySpace>& a,
                                            const Vector<ValueType, MemorySpace>& b,
                                            Vector<ValueType, MemorySpace>& c) {
        thrust::transform(a.begin(), a.end(), b.begin(), c.begin(), thrust::multiplies<ValueType>());
    }

    template <typename ValueType, typename MemorySpace>
    void Vector_element_wise_multiply_Constant(const Vector<ValueType, MemorySpace>& in,
                                            const ValueType& value,
                                            Vector<ValueType, MemorySpace>& out) {
        thrust::transform(in.begin(), in.end(), thrust::make_constant_iterator(value), out.begin(), thrust::multiplies<ValueType>());
    }

}