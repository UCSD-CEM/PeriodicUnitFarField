#include "../include/puff.h"
#include <iostream>
#include <chrono>

using namespace puff;

void benchmark_SparseMatrix_Insertion_Host(int N)
{
    SparseMatrix_h<double> A;
    
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Insertion on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << \
        " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    A.make_matrix();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Make matrix on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << \
        " us" << std::endl;
}

void benchmark_SparseMatrix_Insertion_Device(int N)
{
    SparseMatrix_d<double> A;
    
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Insertion on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << \
        " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    A.make_matrix();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Make matrix on host of size " << N << ": " << \
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << \
        " us" << std::endl;
}


void benchmark_SpMV_Host(int N)
{
    Vector_h<dcomplex> x(N);
    Vector_h<dcomplex> y(N);
    SparseMatrix_h<dcomplex> A;
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, dcomplex(1.0, -1.0));

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
    Vector_d<dcomplex> x(N);
    Vector_d<dcomplex> y(N);
    SparseMatrix_d<dcomplex> A;
    for (int i = 0; i < N; i++)
        A.insert_entry(i, i, dcomplex(1.0, -1.0));

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
    benchmark_SparseMatrix_Insertion_Host(1e8);
    benchmark_SparseMatrix_Insertion_Device(1e8);
    benchmark_SpMV_Host(1e6);
    benchmark_SpMV_Device(1e6);
    return 0;
}