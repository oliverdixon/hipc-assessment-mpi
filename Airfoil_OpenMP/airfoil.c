//
// Created by od641 on 17/11/2025.
//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void perform_work(const int thread_id)
{
    printf("Thread %d has started\n", thread_id);
}

int main()
{
    #pragma omp parallel
    {
        perform_work(omp_get_thread_num());
    }

    return EXIT_SUCCESS;
}
