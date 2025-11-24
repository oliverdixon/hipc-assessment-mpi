//
// Created by od641 on 17/11/2025.
//

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "instance.h"
#include "region.h"

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    const struct instance instance = instance_create();
    struct region region = region_create(&instance);

    instance_describe(&instance, stderr);
    region_describe(&region, stderr);
    region_initialise(&region, &instance);

    char filename[16];
    snprintf(filename, 16, "out/%02d", instance.rank);
    FILE * fp = fopen(filename, "w");
    region_print_flags(&region, fp);
    fclose(fp);

    region_destroy(&region);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
