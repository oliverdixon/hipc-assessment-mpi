//
// Created by od641 on 27/11/2025.
//

#include "exchanger.h"
#include "region.h"

enum tags
{
    TAGS_EMPTY = 0,
    TAGS_NORTH = 1,
    TAGS_SOUTH = 1 << 1,
    TAGS_EAST = 1 << 2,
    TAGS_WEST = 1 << 3
};

enum cell_flags flags_dummy;
compute_t compute_dummy;

static enum tags check_ghost_presence(const struct neighbours * const neighbours)
{
    enum tags tags = TAGS_EMPTY;

    if (neighbours->north != MPI_PROC_NULL)
        tags |= TAGS_NORTH;

    if (neighbours->south != MPI_PROC_NULL)
        tags |= TAGS_SOUTH;

    if (neighbours->east != MPI_PROC_NULL)
        tags |= TAGS_EAST;

    if (neighbours->west != MPI_PROC_NULL)
        tags |= TAGS_WEST;

    return tags;
}

void exchanger_exchange(
    const struct exchanger *const exchanger,
    const struct neighbours *const neighbour_ranks)
{
    MPI_Request requests[4];
    int request_idx = 0;

    // North
    MPI_Isendrecv(
        exchanger->north_row, 1, exchanger->row_t, neighbour_ranks->north, TAGS_NORTH,
        exchanger->north_ghost, 1, exchanger->row_t, neighbour_ranks->north, TAGS_SOUTH,
        exchanger->comm, &requests[request_idx++]);

    // South
    MPI_Isendrecv(
        exchanger->south_row, 1, exchanger->row_t, neighbour_ranks->south, TAGS_SOUTH,
        exchanger->south_ghost, 1, exchanger->row_t, neighbour_ranks->south, TAGS_NORTH,
        exchanger->comm, &requests[request_idx++]);

    // East
    MPI_Isendrecv(
        exchanger->east_col, 1, exchanger->col_t, neighbour_ranks->east, TAGS_EAST,
        exchanger->east_ghost, 1, exchanger->col_t, neighbour_ranks->east, TAGS_WEST,
        exchanger->comm, &requests[request_idx++]);

    // West
    MPI_Isendrecv(
        exchanger->west_col, 1, exchanger->col_t, neighbour_ranks->west, TAGS_WEST,
        exchanger->west_ghost, 1, exchanger->col_t, neighbour_ranks->west, TAGS_EAST,
        exchanger->comm, &requests[request_idx++]);

    MPI_Waitall(request_idx, requests, MPI_STATUSES_IGNORE);
}

struct exchanger exchanger_create_flags(
    enum cell_flags *const * const data,
    const struct iterator h_bounds,
    const struct iterator v_bounds,
    const MPI_Comm comm, // NOLINT(*-misplaced-const)
    const MPI_Datatype row_t, // NOLINT(*-misplaced-const)
    const MPI_Datatype col_t, // NOLINT(*-misplaced-const)
    const struct neighbours * const neighbours)
{
    const indexer_t start_h_idx = h_bounds.begin;
    const enum tags ghosts = check_ghost_presence(neighbours);
    const struct exchanger exchanger = {
        .row_t = row_t,
        .col_t = col_t,
        .comm = comm,

        .north_row = &data[start_h_idx][v_bounds.begin],
        .north_ghost = ghosts & TAGS_NORTH ? &data[start_h_idx][v_bounds.begin - 1] : &flags_dummy,

        .south_row = &data[start_h_idx][v_bounds.end - 1],
        .south_ghost = ghosts & TAGS_SOUTH ? &data[start_h_idx][v_bounds.end] : &flags_dummy,

        .west_col = data[start_h_idx],
        .west_ghost = ghosts & TAGS_WEST ? data[start_h_idx - 1] : &flags_dummy,

        .east_col = data[h_bounds.end - 1],
        .east_ghost = ghosts & TAGS_EAST ? data[h_bounds.end] : &flags_dummy
    };

    return exchanger;
}

struct exchanger exchanger_create_compute(
    compute_t *const * const data,
    const struct iterator h_bounds,
    const struct iterator v_bounds,
    const MPI_Comm comm, // NOLINT(*-misplaced-const)
    const MPI_Datatype row_t, // NOLINT(*-misplaced-const)
    const MPI_Datatype col_t, // NOLINT(*-misplaced-const)
    const struct neighbours * const neighbours)
{
    const indexer_t start_h_idx = h_bounds.begin;
    const enum tags ghosts = check_ghost_presence(neighbours);
    const struct exchanger exchanger = {
        .row_t = row_t,
        .col_t = col_t,
        .comm = comm,

        .north_row = &data[start_h_idx][v_bounds.begin],
        .north_ghost = ghosts & TAGS_NORTH ? &data[start_h_idx][v_bounds.begin - 1] : &compute_dummy,

        .south_row = &data[start_h_idx][v_bounds.end - 1],
        .south_ghost = ghosts & TAGS_SOUTH ? &data[start_h_idx][v_bounds.end] : &compute_dummy,

        .west_col = data[start_h_idx],
        .west_ghost = ghosts & TAGS_WEST ? data[start_h_idx - 1] : &compute_dummy,

        .east_col = data[h_bounds.end - 1],
        .east_ghost = ghosts & TAGS_EAST ? data[h_bounds.end] : &compute_dummy
    };

    return exchanger;
}
