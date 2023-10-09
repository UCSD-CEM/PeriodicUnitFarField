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
    puff::Vector_h<puff::complex_d> a(N);
    puff::Vector_h<puff::complex_d> b(N);
    puff::Vector_h<puff::complex_d> c(N);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
	{
        a[i] = puff::complex_d(i, i);
        b[i] = puff::complex_d(i, -i);
	}

    puff::Vector_element_wise_multiply_Constant(a, puff::complex_d(3.0, 0.0), c);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(c[i], puff::complex_d(3.0 * i, 3.0 * i));
    }

    puff::Vector_element_wise_multiply_Vector(a, b, c);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(c[i], puff::complex_d(2.0 * i * i, 0.0));
    }
}

TEST(PUFF, Check_Complex_Array_Device)
{
    puff::Vector_d<puff::complex_d> d_a(N);
    puff::Vector_d<puff::complex_d> d_b(N);
    puff::Vector_d<puff::complex_d> d_c(N);
    puff::Vector_h<puff::complex_d> h_a(N);
    puff::Vector_h<puff::complex_d> h_b(N);
    puff::Vector_h<puff::complex_d> h_c(N);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        h_a[i] = puff::complex_d(i, i);
        h_b[i] = puff::complex_d(i, -i);
    }
    d_a = h_a;
    d_b = h_b;

    puff::Vector_element_wise_multiply_Constant(d_a, puff::complex_d(3.0, 0.0), d_c);
    h_c = d_c;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(h_c[i], puff::complex_d(3.0 * i, 3.0 * i));
    }

    puff::Vector_element_wise_multiply_Vector(d_a, d_b, d_c);
    h_c = d_c;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(h_c[i], puff::complex_d(2.0 * i * i, 0.0));
    }
}