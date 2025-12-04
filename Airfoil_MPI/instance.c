//
// Created by od641 on 18/11/2025.
//

#include "instance.h"

#include <stdlib.h>

#include "region.h"

static MPI_Datatype create_dim2_type()
{
    static const int field_count = 2;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefault-const-init-var-unsafe"
    const struct dim2 dim2_t_dummy;
#pragma clang diagnostic pop

    MPI_Aint dim2_t_base_address;
    MPI_Get_address(&dim2_t_dummy, &dim2_t_base_address);

    MPI_Aint dim2_t_displacements[field_count];
    MPI_Get_address(&dim2_t_dummy.x, &dim2_t_displacements[0]);
    MPI_Get_address(&dim2_t_dummy.y, &dim2_t_displacements[1]);
    dim2_t_displacements[0] = MPI_Aint_diff(dim2_t_displacements[0], dim2_t_base_address);
    dim2_t_displacements[1] = MPI_Aint_diff(dim2_t_displacements[1], dim2_t_base_address);

    const int dim2_t_lengths[] = { 1, 1 };
    const MPI_Datatype dim2_t_types[] = { MPI_UNSIGNED, MPI_UNSIGNED };

    MPI_Datatype dim2_t;
    MPI_Type_create_struct(field_count, dim2_t_lengths, dim2_t_displacements, dim2_t_types, &dim2_t);
    MPI_Type_commit(&dim2_t);

    return dim2_t;
}

static void collect_dimension_data(const struct instance *const instance, const struct region *const region,
    struct dim2 * const indents, indexer_t * const widths, indexer_t * const heights)
{
    MPI_Request reqs[3];
    int request_idx = 0;

    const struct dim2 shifted_indents = instance_translate_to_cells(instance, &region->indents);
    MPI_Igather(&shifted_indents, 1, instance->dim2_t, indents, 1, instance->dim2_t, 0, instance->cartesian_comm,
        &reqs[request_idx++]);

    const indexer_t width = region->h_exterior.end - region->h_exterior.begin;
    MPI_Igather(&width, 1, MPI_UNSIGNED, widths, 1, MPI_UNSIGNED, 0, instance->cartesian_comm, &reqs[request_idx++]);

    const indexer_t height = region->v_exterior.end - region->v_exterior.begin;
    MPI_Igather(&height, 1, MPI_UNSIGNED, heights, 1, MPI_UNSIGNED, 0, instance->cartesian_comm, &reqs[request_idx++]);

    MPI_Waitall(request_idx, reqs, MPI_STATUSES_IGNORE);
}

struct instance instance_create()
{
    int rank;
    int count;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &count);

    int dims[] = {0, 0};
    const int periods[] = {0, 0};
    MPI_Comm cartesian_comm;

    MPI_Dims_create(count, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartesian_comm);

    int coords[2];
    MPI_Cart_coords(cartesian_comm, rank, 2, coords);

    struct instance instance = {
        .rank = rank,
        .count = count,
        .cartesian_comm = cartesian_comm,

        .dim_extents.x = dims[0],
        .dim_extents.y = dims[1],

        .cartesian_pos.x = coords[0],
        .cartesian_pos.y = coords[1],

        .problem_size.x = 4.0,
        .problem_size.y = 1.0,

        .naca_specifier = {
            .maximum_camber = 2,
            .edge_distance = 4,
            .maximum_thickness = 12
        },
        .dim2_t = create_dim2_type()
    };

    MPI_Cart_shift(cartesian_comm, 0, 1, &instance.neighbours.west, &instance.neighbours.east);
    MPI_Cart_shift(cartesian_comm, 1, 1, &instance.neighbours.north, &instance.neighbours.south);

    return instance;
}

void instance_destroy(struct instance *instance)
{
    MPI_Type_free(&instance->dim2_t);
    MPI_Comm_free(&instance->cartesian_comm);
}

void instance_describe(const struct instance *instance, FILE *const destination)
{
    fprintf(destination,
            "Instance statistics:\n\t"
            "Rank: %d / %d\n\t"
            "Dimensions: (%d, %d)\n\t"
            "Cartesian co-ordinates: (%d, %d)\n\t"
            "Global problem size: (%lf, %lf)\n\t"
            "NACA specifier: %2d%1d%1d\n",

            instance->rank, instance->count - 1, instance->dim_extents.x, instance->dim_extents.y,
            instance->cartesian_pos.x, instance->cartesian_pos.y, instance->problem_size.x, instance->problem_size.y,
            instance->naca_specifier.maximum_camber, instance->naca_specifier.edge_distance,
            instance->naca_specifier.maximum_thickness);
}

void instance_serialise_vtk(
    const struct instance *const instance,
    const struct region *const region,
    const unsigned int max_subfile_digits,
    const char * const subfile_prefix,
    const char * const subfile_extension,
    FILE *const destination)
{
    struct dim2 * shifted_indents = NULL;
    indexer_t * widths = NULL;
    indexer_t * heights = NULL;

    if (instance->rank == 0) {
        shifted_indents = malloc(sizeof(struct dim2) * instance->count);
        widths = malloc(sizeof(indexer_t) * instance->count);
        heights = malloc(sizeof(indexer_t) * instance->count);
    }

    collect_dimension_data(instance, region, shifted_indents, widths, heights);
    if (instance->rank != 0)
        return;

    fprintf(destination,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "\t<PRectilinearGrid WholeExtent=\"0 %u 0 %u 0 0\" GhostLevel=\"0\">\n"
        "\t\t<PCoordinates>\n"
        "\t\t\t<PDataArray type=\"Float64\" />\n"
        "\t\t\t<PDataArray type=\"Float64\" />\n"
        "\t\t\t<PDataArray type=\"Float64\" />\n"
        "\t\t</PCoordinates>\n"
        "\t\t<PPointData Vectors=\"uv\">\n"
        "\t\t\t<PDataArray type=\"Float64\" Name=\"uv\" NumberOfComponents=\"3\" />\n"
        "\t\t</PPointData>\n"
        "\t\t<PCellData Scalars=\"p\">\n"
        "\t\t\t<PDataArray type=\"Float64\" Name=\"p\" />\n"
        "\t\t</PCellData>\n",

        (unsigned int) instance->problem_size.x * region->resolution,
        (unsigned int) instance->problem_size.y * region->resolution);

    for (unsigned int rank_id = 0; rank_id < instance->count; ++rank_id)
        fprintf(destination,
            "\t\t<Piece Extent=\"%u %u %u %u 0 0\" Source=\"%s%0*d%s\" />\n",
            shifted_indents[rank_id].x, shifted_indents[rank_id].x + widths[rank_id] - 1,
            shifted_indents[rank_id].y, shifted_indents[rank_id].y + heights[rank_id] - 1,
            subfile_prefix, max_subfile_digits, rank_id, subfile_extension);

    fputs("\t</PRectilinearGrid>\n"
          "</VTKFile>\n", destination);

    free(heights);
    free(widths);
    free(shifted_indents);
}

struct dim2 instance_translate_to_cells(const struct instance *instance, const struct dim2 *points_source)
{
    const struct dim2 v = {
        .x = points_source->x - instance->cartesian_pos.x,
        .y = points_source->y - instance->cartesian_pos.y
    };

    return v;
}

struct dim2 instance_translate_to_points(const struct instance *instance, const struct dim2 *cell_source)
{
    const struct dim2 v = {
        .x = cell_source->x + instance->cartesian_pos.x,
        .y = cell_source->y + instance->cartesian_pos.y
    };

    return v;
}

struct dim2 instance_get_indentations(const struct instance * const instance, const struct dim2 own_size)
{
    struct dim2 indents;
    const struct dim2 size = {
        .x = instance->cartesian_pos.x == 0 ? 0 : own_size.x,
        .y = instance->cartesian_pos.y == 0 ? 0 : own_size.y
    };

    int fix_dimensions[] = {1, 0};
    MPI_Comm fixed_dim_comm;
    MPI_Cart_sub(instance->cartesian_comm, fix_dimensions, &fixed_dim_comm); // Fixed on X; row communicator.
    MPI_Scan(&size.x, &indents.x, 1, MPI_UNSIGNED, MPI_SUM, fixed_dim_comm);

    fix_dimensions[0] = 0;
    fix_dimensions[1] = 1;
    MPI_Cart_sub(instance->cartesian_comm, fix_dimensions, &fixed_dim_comm); // Fixed on Y; column communicator.
    MPI_Scan(&size.y, &indents.y, 1, MPI_UNSIGNED, MPI_SUM, fixed_dim_comm);

    return indents;
}
