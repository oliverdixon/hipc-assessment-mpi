//
// Created by od641 on 17/11/2025.
//

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "airfoil.h"

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int slot_id;
    int slot_count;

    MPI_Comm_rank(MPI_COMM_WORLD, &slot_id);
    MPI_Comm_size(MPI_COMM_WORLD, &slot_count);

    printf("Slot %d out of %d.\n", slot_id + 1, slot_count);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
