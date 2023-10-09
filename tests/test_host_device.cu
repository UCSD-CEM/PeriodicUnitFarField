#include "../include/puff.h"
#include <gtest/gtest.h>
#include <omp.h>

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