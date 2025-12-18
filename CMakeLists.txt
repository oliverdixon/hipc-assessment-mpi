cmake_minimum_required(VERSION 4.0)
project(HIPC_Assessment_MPI LANGUAGES C)

include(CheckCCompilerFlag)
check_c_compiler_flag(-Wunknown-warning-option WARN_UNKNOWN_WARNING)
if (WARN_UNKNOWN_WARNING)
    add_compile_options(-Wno-unknown-warning-option)
endif()

find_package(MPI REQUIRED COMPONENTS C)

add_executable(Airfoil_MPI
        Source/airfoil.c
        Source/region.c
        Source/region.h
        Source/instance.c
        Source/instance.h
        Source/types.h
)

set_property(TARGET Airfoil_MPI PROPERTY CMAKE_C_STANDARD 11)
target_link_libraries(Airfoil_MPI PRIVATE MPI::MPI_C m)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    target_link_options(Airfoil_MPI PRIVATE -fuse-ld=lld)
    target_compile_options(Airfoil_MPI PRIVATE
            -axCORE-AVX2
            -fast
            -Rpass=loop-vectorize
            -Rpass-analysis=loop-vectorize
            -Rpass-missed=loop-vectorize
            -fsave-optimization-record
    )
endif ()
