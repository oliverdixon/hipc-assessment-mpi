cmake_minimum_required(VERSION 4.0)
project(HIPC_Assessment LANGUAGES C)

add_executable(Airfoil_Serial
        Airfoil_Serial/airfoil.c
        Airfoil_Serial/args.c
        Airfoil_Serial/args.h
        Airfoil_Serial/boundary.c
        Airfoil_Serial/boundary.h
        Airfoil_Serial/data.c
        Airfoil_Serial/data.h
        Airfoil_Serial/setup.c
        Airfoil_Serial/setup.h
        Airfoil_Serial/vtk.c
        Airfoil_Serial/vtk.h)

set_property(TARGET Airfoil_Serial PROPERTY CMAKE_C_STANDARD 11)
target_link_libraries(Airfoil_Serial PRIVATE m)
