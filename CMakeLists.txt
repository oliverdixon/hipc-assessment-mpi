cmake_minimum_required(VERSION 4.0)
project(HIPC_Assessment LANGUAGES C)
include(CheckCCompilerFlag)

# Serial

add_executable(Airfoil_Serial
        Airfoil_Serial/airfoil.c
        Airfoil_Serial/args.c
        Airfoil_Serial/args.h
        Airfoil_Serial/data.c
        Airfoil_Serial/data.h
        Airfoil_Serial/setup.c
        Airfoil_Serial/setup.h
        Airfoil_Serial/vtk.c
        Airfoil_Serial/vtk.h
)

set_property(TARGET Airfoil_Serial PROPERTY CMAKE_C_STANDARD 11)
target_link_libraries(Airfoil_Serial PRIVATE m)

check_c_compiler_flag(-Wc23-extensions WARN_C23_EXTENSIONS)
if (WARN_C23_EXTENSIONS)
    target_compile_options(Airfoil_Serial PRIVATE -Wno-c23-extensions)
endif()

# OpenMP

find_package(OpenMP REQUIRED COMPONENTS C)

add_executable(Airfoil_OpenMP
        Airfoil_OpenMP/airfoil.c
        Airfoil_OpenMP/airfoil.h
)

target_link_libraries(Airfoil_OpenMP PRIVATE OpenMP::OpenMP_C)

# MPI

find_package(MPI REQUIRED COMPONENTS C)

add_executable(Airfoil_MPI
        Airfoil_MPI/airfoil.c
        Airfoil_MPI/airfoil.h
        Airfoil_MPI/region.c
        Airfoil_MPI/region.h
        Airfoil_MPI/instance.c
        Airfoil_MPI/instance.h
)

set_property(TARGET Airfoil_MPI PROPERTY CMAKE_C_STANDARD 11)
target_link_libraries(Airfoil_MPI PRIVATE MPI::MPI_C m)

# CUDA

option(BUILD_AIRFOIL_CUDA "Build the Airfoil CUDA variant.")

if (BUILD_AIRFOIL_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(Airfoil_CUDA
            Airfoil_CUDA/airfoil.cu
    )

    set_property(TARGET Airfoil_CUDA PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()
