//
// Created by od641 on 10/12/2025.
//

#ifndef HIPC_ASSESSMENT_TYPES_H
#define HIPC_ASSESSMENT_TYPES_H

#include <cassert>
#include <cstdio>

typedef double compute_t;
typedef unsigned int indexer_t;

struct iterator
{
    indexer_t begin;
    indexer_t end;
};

struct dim2
{
    indexer_t x;
    indexer_t y;
};

struct compute_dim2
{
    compute_t x;
    compute_t y;
};

#ifndef TOSTRING
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#endif

#define safe_cuda(expr) \
    do { \
        cudaError_t status = expr; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "%s: expression %s failed with error %s\n", \
                __FILE__ ":" TOSTRING(__LINE__), \
                #expr, \
                cudaGetErrorString(status)); \
            assert(0); \
        } \
    } while (0)

#endif // HIPC_ASSESSMENT_TYPES_H
