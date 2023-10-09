#pragma once
// If OpenMP is available, use it
#ifdef USE_OPENMP
#include <omp.h>
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP
#endif
#include "SparseMatrix.h"
#include "FFT_Conv3D.h"

namespace puff {
	using complex_d = thrust::complex<double>;
	using complex_f = thrust::complex<float>;
}
