//
// Created by od641 on 17/11/2025.
//

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "instance.h"
#include "region.h"

static void serialise(const struct instance * const instance, const struct region * const region)
{
    static const unsigned int max_rank_digits = 2;
    assert(instance->count < (int) powf(10, max_rank_digits));

    static const char prefix[] = "flows";
    static const char suffix[] = "_00.vtr";

    chdir("./out/");

    FILE * const master_fp = instance->rank == 0 ? fopen("flows.pvtr", "w") : NULL;
    instance_serialise_vtk(instance, region, max_rank_digits, prefix, suffix, master_fp);

    if (master_fp != NULL)
        fclose(master_fp);

    static unsigned int prefix_length = sizeof(prefix) / sizeof(*prefix) - 1;
    static unsigned int suffix_length = sizeof(suffix) / sizeof(*suffix) - 1;

    char subfile_name[prefix_length + max_rank_digits + suffix_length + 1];
    sprintf(subfile_name, "%s%0*d%s", prefix, max_rank_digits, instance->rank, suffix);
    FILE * const subfile_fp = fopen(subfile_name, "w");
    region_serialise_vtk(region, instance, subfile_fp);
    fclose(subfile_fp);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    struct instance instance = instance_create();
    struct region region = region_create(&instance);

    instance_describe(&instance, stderr);

    region_describe(&region, stderr);
    region_initialise(&region, &instance);

    region_halo_exchange(&region, &instance);
    region_apply_boundary_conditions(&region);
    region_halo_exchange(&region, &instance);
    step(&region);

    serialise(&instance, &region);

    region_destroy(&region);
    instance_destroy(&instance);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
