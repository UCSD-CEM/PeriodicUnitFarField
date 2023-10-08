#pragma once
// If OpenMP is available, use it
#ifdef USE_OPENMP
#include <omp.h>
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP
#endif
#include "SparseMatrix.h"
#include "FFT_Conv3D.h"
typedef thrust::complex<double> complex_d;
typedef thrust::complex<float> complex_f;

namespace puff {

}
