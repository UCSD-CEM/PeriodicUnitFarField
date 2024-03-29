#include "../include/puff.h"
#include <gtest/gtest.h>
#include <omp.h>

static constexpr int N = 1e6;

TEST(PUFF, init)
{
#ifdef USE_OPENMP
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "**********GTest initialized with OpenMP**********" << std::endl;
#else
    std::cout << "**********GTest initialized without OpenMP**********" << std::endl;
#endif
}

TEST(PUFF, Check_OMP)
{
#ifdef USE_OPENMP
#pragma omp parallel
    {
        EXPECT_EQ(omp_get_num_threads(), omp_get_max_threads());
    }  
#else
    ASSERT_TRUE(true);
#endif
}

TEST(PUFF, Check_Complex_Array_Host)
{
    puff::Vector_h<puff::dcomplex> a(N);
    puff::Vector_h<puff::dcomplex> b(N);
    puff::Vector_h<puff::dcomplex> c(N);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
	{
        a[i] = puff::dcomplex(i, i);
        b[i] = puff::dcomplex(i, -i);
	}

    puff::Vector_element_wise_multiply_Constant(a, puff::dcomplex(3.0, 0.0), c);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(c[i], puff::dcomplex(3.0 * i, 3.0 * i));
    }

    puff::Vector_element_wise_multiply_Vector(a, b, c);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(c[i], puff::dcomplex(2.0 * i * i, 0.0));
    }
}

TEST(PUFF, Check_Complex_Array_Device)
{
    puff::Vector_d<puff::dcomplex> d_a(N);
    puff::Vector_d<puff::dcomplex> d_b(N);
    puff::Vector_d<puff::dcomplex> d_c(N);
    puff::Vector_h<puff::dcomplex> h_a(N);
    puff::Vector_h<puff::dcomplex> h_b(N);
    puff::Vector_h<puff::dcomplex> h_c(N);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        h_a[i] = puff::dcomplex(i, i);
        h_b[i] = puff::dcomplex(i, -i);
    }
    d_a = h_a;
    d_b = h_b;

    puff::Vector_element_wise_multiply_Constant(d_a, puff::dcomplex(3.0, 0.0), d_c);
    h_c = d_c;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(h_c[i], puff::dcomplex(3.0 * i, 3.0 * i));
    }

    puff::Vector_element_wise_multiply_Vector(d_a, d_b, d_c);
    h_c = d_c;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(h_c[i], puff::dcomplex(2.0 * i * i, 0.0));
    }
}

TEST(PUFF, Check_Complex_SpMV_Host)
{
    puff::Vector_h<puff::dcomplex> x(N);
    puff::Vector_h<puff::dcomplex> y(N);
    puff::SparseMatrix_h<puff::dcomplex> A;

    for(int i = 0; i < N; i++)
    {
        x[i] = puff::dcomplex(1, 1);
        A.insert_entry(i, N - 1 - i, puff::dcomplex(i, -i));
    }
    A.make_matrix();

    // Multiply y = A * x
    A.SpMV(x, y);
    for(int i = 0; i < N; i++)
    {
        EXPECT_EQ(y[i], puff::dcomplex(2.0 * i, 0.0));
    }

    // Conjugate multiply y = A^H * x
    A.SpMV(x, y, false, true);
    for(int i = 0; i < N; i++)
    {
        EXPECT_EQ(y[i], puff::dcomplex(0.0, 2.0 * i));
    }

    // Transpose multiply y = A' * x
    A.SpMV(x, y, true);
    for(int i = 0; i < N; i++)
    {
       EXPECT_EQ(y[i], puff::dcomplex(2.0 * (N - 1 - i), 0.0));
    }

    // Conjugate transpose multiply y = A^H * x
    A.SpMV(x, y, true, true);
    for(int i = 0; i < N; i++)
    {
       EXPECT_EQ(y[i], puff::dcomplex(0.0, 2.0 * (N - 1 - i)));
    }

}

TEST(PUFF, Check_Complex_SpMV_Device)
{
    puff::Vector_h<puff::dcomplex> h_x(N);
    puff::Vector_h<puff::dcomplex> h_y(N);

    puff::Vector_d<puff::dcomplex> d_x(N);
    puff::Vector_d<puff::dcomplex> d_y(N);
    puff::SparseMatrix_d<puff::dcomplex> A;

    for(int i = 0; i < N; i++)
    {
        h_x[i] = puff::dcomplex(1, 1);
        A.insert_entry(i, N - 1 - i, puff::dcomplex(i, -i));
    }
    A.make_matrix();
    d_x = h_x;
    // Multiply y = A * x
    A.SpMV(d_x, d_y);
    h_y = d_y;
    for(int i = 0; i < N; i++)
    {
        EXPECT_EQ(h_y[i], puff::dcomplex(2.0 * i, 0.0));
    }

    // Conjugate multiply y = A^H * x
    A.SpMV(d_x, d_y, false, true);
    h_y = d_y;
    for(int i = 0; i < N; i++)
    {
        EXPECT_EQ(h_y[i], puff::dcomplex(0.0, 2.0 * i));
    }

    // Transpose multiply y = A' * x
    A.SpMV(d_x, d_y, true);
    h_y = d_y;
    for(int i = 0; i < N; i++)
    {
       EXPECT_EQ(h_y[i], puff::dcomplex(2.0 * (N - 1 - i), 0.0));
    }

    // Conjugate transpose multiply y = A^H * x
    A.SpMV(d_x, d_y, true, true);
    h_y = d_y;
    for(int i = 0; i < N; i++)
    {
       EXPECT_EQ(h_y[i], puff::dcomplex(0.0, 2.0 * (N - 1 - i)));
    }

}

TEST(PUFF, Check_spectral_radius_Host)
{
    puff::SparseMatrix_h<double> A;
    for(int i = 0; i < N; i++)
        A.insert_entry(i, i, i + 1);

    A.make_matrix();

    double rho = N ;
    EXPECT_TRUE((std::abs(A.spectral_radius(100, true) - rho) / rho) < 0.01); // force using lanczos on purpose
}

TEST(PUFF, Check_spectral_radius_Device)
{
    puff::SparseMatrix_d<double> A;
    for(int i = 0; i < N; i++)
        A.insert_entry(i, i, i + 1);
    
    A.make_matrix();

    double rho = N;
    EXPECT_TRUE((std::abs(A.spectral_radius(100, false) - rho) / rho) < 0.01); // force using arnoldi on purpose
}

TEST(PUFF, Check_complex_gmres_host)
{
    puff::Vector_h<puff::dcomplex> x(N, puff::dcomplex(1, 1));
    puff::Vector_h<puff::dcomplex> b(N);
    puff::SparseMatrix_h<puff::dcomplex> A;

    for(int i = 0; i < N; i++)
        A.insert_entry(i, i, puff::dcomplex(i + N / 10, -(i + N / 10)));

    A.make_matrix();

    // Multiply b = A * x
    A.SpMV(x, b);

    // Set initial guess to zero
    thrust::fill(x.begin(), x.end(), puff::dcomplex(0.0, 0.0));

    // Solve Ax = b
    A.gmres(x, b);

    for(int i = 0; i < N; i++)
    {
        EXPECT_NEAR(x[i].real(), 1.0, 1e-3);
        EXPECT_NEAR(x[i].imag(), 1.0, 1e-3);
    }
}

TEST(PUFF, Check_complex_gmres_device)
{
    puff::Vector_h<puff::dcomplex> h_x(N);
    puff::Vector_d<puff::dcomplex> d_x(N, puff::dcomplex(1, 1));
    puff::Vector_d<puff::dcomplex> d_b(N);
    puff::SparseMatrix_d<puff::dcomplex> A;

    for(int i = 0; i < N; i++)
        A.insert_entry(i, i, puff::dcomplex(i + N / 10, -(i + N / 10)));

    A.make_matrix();

    // Multiply b = A * x
    A.SpMV(d_x, d_b);

    // Set initial guess to zero
    thrust::fill(d_x.begin(), d_x.end(), puff::dcomplex(0.0, 0.0));

    // Solve Ax = b
    A.gmres(d_x, d_b);
    h_x = d_x;

    for(int i = 0; i < N; i++)
    {
        EXPECT_NEAR(h_x[i].real(), 1.0, 1e-3);
        EXPECT_NEAR(h_x[i].imag(), 1.0, 1e-3);
    }
}