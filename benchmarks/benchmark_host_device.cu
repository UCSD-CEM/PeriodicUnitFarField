#include "../include/puff.h"
#include <iostream>
#include <chrono>

using namespace puff;

void benchmark_SpMV_Host(int N)
{
    Vector_h<complex_d> x(N);
    Vector_h<complex_d> y(N);
    SparseMatrix_h<complex_d> A;
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, complex_d(1.0, -1.0));

    A.make_matrix();

    // Warmup
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y);
    
    // Benchmark y = A * x
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 100; i++)
		A.SpMV(x, y);
    
    auto end = std::chrono::high_resolution_clock::now();
    // output in microseconds
    std::cout << "SpMV on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100 << \
        " us" << std::endl;

    // warmup
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y, true);

    // Benchmark y = A' * x
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y, true);
    
	end = std::chrono::high_resolution_clock::now();
	// output in microseconds
	std::cout << "Transpose SpMV on host of size " << N << ": " << \
		std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100 << \
		" us" << std::endl;
    return;
}

void benchmark_SpMV_Device(int N)
{
    Vector_d<complex_d> x(N);
    Vector_d<complex_d> y(N);
    SparseMatrix_d<complex_d> A;
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, complex_d(1.0, -1.0));

    A.make_matrix();

    // Warmup
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y);

    // Benchmark y = A * x
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++)
        A.SpMV(x, y);

    auto end = std::chrono::high_resolution_clock::now();
    // output in microseconds
    std::cout << "SpMV on device of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100 << \
        " us" << std::endl;


    // warmup
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y, true);

    // Benchmark y = A' * x
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++)
        A.SpMV(x, y, true);

    end = std::chrono::high_resolution_clock::now();
    // output in microseconds
    std::cout << "Transpose SpMV on device of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100 << \
        " us" << std::endl;
    return;
}


int main()
{
#ifdef USE_OPENMP
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "**********Benchmark initialized with OpenMP**********" << std::endl;
#else
    std::cout << "**********Benchmark initialized without OpenMP**********" << std::endl;
#endif
    benchmark_SpMV_Host(1e6);
    benchmark_SpMV_Device(1e6);
    return 0;
}