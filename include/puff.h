#pragma once
// If OpenMP is available, use it
#ifdef USE_OPENMP
#include <omp.h>
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP
#endif
#include "SparseMatrix.h"
#include "CConv3D.h"

namespace puff {
	using dcomplex = thrust::complex<double>;
	using fcomplex = thrust::complex<float>;
}
