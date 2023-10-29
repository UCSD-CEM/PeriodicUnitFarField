
#pragma once

#include "SparseMatrix.h"
#include "mkl.h"
#include "mkl_dfti.h"
#include "mkl_service.h"
#include "cufft.h"
#include "cuda_runtime.h"
#include "utils.h"


namespace puff{


template<typename ValueType, typename MemorySpace> // Derived class
class FFT3D{
    FFT3D() {}
    ~FFT3D() {}
};


template<typename ValueType, typename MemorySpace> // Derived class
class CCONV3D{
    public:
        CCONV3D() {}
        ~CCONV3D() {}

    private:
        FFT3D<ValueType, MemorySpace> fft3d;
};

template<typename ValueType>
using FFT3D_h = FFT3D<ValueType, cusp::host_memory>;

template<typename ValueType>
using FFT3D_d = FFT3D<ValueType, cusp::device_memory>;

template<typename ValueType>
using CCONV3D_h = CCONV3D<ValueType, cusp::host_memory>;

template<typename ValueType>
using CCONV3D_d = CCONV3D<ValueType, cusp::device_memory>;

}

#include "details/CConv3D.inl"