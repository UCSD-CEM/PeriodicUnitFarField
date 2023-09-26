#include "../include/puff.h"

int main()
{
#ifdef USE_OPENMP
    omp_set_num_threads(omp_get_max_threads());
#endif
    return 0;
}