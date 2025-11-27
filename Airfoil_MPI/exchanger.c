//
// Created by od641 on 27/11/2025.
//

#include "exchanger.h"
#include "region.h"

enum tags
{
    NORTH_ORIGIN,
    SOUTH_ORIGIN,
    EAST_ORIGIN,
    WEST_ORIGIN
};

void exchanger_exchange(
    const struct exchanger *const exchanger,
    const struct neighbours *const neighbour_ranks)
{
    MPI_Request requests[4];
    int request_idx = 0;

    // North
    MPI_Isendrecv(
        exchanger->north_row, 1, exchanger->row_t, neighbour_ranks->north, NORTH_ORIGIN,
        exchanger->north_ghost, 1, exchanger->row_t, neighbour_ranks->north, SOUTH_ORIGIN,
        exchanger->comm, &requests[request_idx++]);

    // South
    MPI_Isendrecv(
        exchanger->south_row, 1, exchanger->row_t, neighbour_ranks->south, SOUTH_ORIGIN,
        exchanger->south_ghost, 1, exchanger->row_t, neighbour_ranks->south, NORTH_ORIGIN,
        exchanger->comm, &requests[request_idx++]);

    // East
    MPI_Isendrecv(
        exchanger->east_col, 1, exchanger->col_t, neighbour_ranks->east, EAST_ORIGIN,
        exchanger->east_ghost, 1, exchanger->col_t, neighbour_ranks->east, WEST_ORIGIN,
        exchanger->comm, &requests[request_idx++]);

    // West
    MPI_Isendrecv(
        exchanger->west_col, 1, exchanger->col_t, neighbour_ranks->west, WEST_ORIGIN,
        exchanger->west_ghost, 1, exchanger->col_t, neighbour_ranks->west, EAST_ORIGIN,
        exchanger->comm, &requests[request_idx++]);

    MPI_Waitall(request_idx, requests, MPI_STATUSES_IGNORE);
}

struct exchanger exchanger_create_flags(
    enum cell_flags *const * const data,
    const struct iterator h_bounds,
    const struct iterator v_bounds,
    const MPI_Comm comm, // NOLINT(*-misplaced-const)
    const MPI_Datatype row_t, // NOLINT(*-misplaced-const)
    const MPI_Datatype col_t) // NOLINT(*-misplaced-const)
{
    const indexer_t start_h_idx = h_bounds.begin - 1;
    const struct exchanger exchanger = {
        .row_t = row_t,
        .col_t = col_t,
        .comm = comm,

        .north_row = &data[start_h_idx][v_bounds.begin],
        .north_ghost = &data[start_h_idx][v_bounds.begin - 1],
        .south_row = &data[start_h_idx][v_bounds.end - 1],
        .south_ghost = &data[start_h_idx][v_bounds.end],
        .west_col = data[h_bounds.begin],
        .west_ghost = data[h_bounds.begin - 1],
        .east_col = data[h_bounds.end - 1],
        .east_ghost = data[h_bounds.end]
    };

    return exchanger;
}

struct exchanger exchanger_create_compute(
    compute_t *const * const data,
    const struct iterator h_bounds,
    const struct iterator v_bounds,
    const MPI_Comm comm, // NOLINT(*-misplaced-const)
    const MPI_Datatype row_t, // NOLINT(*-misplaced-const)
    const MPI_Datatype col_t) // NOLINT(*-misplaced-const)
{
    const indexer_t start_h_idx = h_bounds.begin - 1;
    const struct exchanger exchanger = {
        .row_t = row_t,
        .col_t = col_t,
        .comm = comm,

        .north_row = &data[start_h_idx][v_bounds.begin],
        .north_ghost = &data[start_h_idx][v_bounds.begin - 1],
        .south_row = &data[start_h_idx][v_bounds.end - 1],
        .south_ghost = &data[start_h_idx][v_bounds.end],
        .west_col = data[h_bounds.begin],
        .west_ghost = data[h_bounds.begin - 1],
        .east_col = data[h_bounds.end - 1],
        .east_ghost = data[h_bounds.end]
    };

    return exchanger;
}