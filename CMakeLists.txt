cmake_minimum_required(VERSION 3.16)
project(SparseSolv VERSION 2.1.0 LANGUAGES CXX)

# Header-only library
add_library(sparsesolv INTERFACE)
target_include_directories(sparsesolv INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(sparsesolv INTERFACE cxx_std_17)

# OpenMP (optional)
option(SPARSESOLV_USE_OPENMP "Use OpenMP for parallelization" ON)
if(SPARSESOLV_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(sparsesolv INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

# Alias for consistent naming
add_library(SparseSolv::sparsesolv ALIAS sparsesolv)

# ==============================================================================
# NGSolve Python extension module (optional)
# ==============================================================================
option(SPARSESOLV_BUILD_NGSOLVE "Build NGSolve Python extension module" OFF)

if(SPARSESOLV_BUILD_NGSOLVE)
    if(NOT NGSOLVE_INSTALL_DIR)
        execute_process(
            COMMAND python -c "import ngsolve, pathlib; d = pathlib.Path(ngsolve.__file__).parent.parent; print(d.as_posix() + ';' + (d.parent.parent / 'cmake').as_posix() + ';' + (d / 'cmake').as_posix())"
            OUTPUT_VARIABLE NGSOLVE_HINTS
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    find_package(NGSolve CONFIG REQUIRED
        HINTS ${NGSOLVE_HINTS} "${NGSOLVE_INSTALL_DIR}"
    )
    message(STATUS "Found NGSolve: ${NGSOLVE_INCLUDE_DIRS}")

    add_ngsolve_python_module(sparsesolv_ngsolve ngsolve/python_module.cpp)
    target_include_directories(sparsesolv_ngsolve PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(sparsesolv_ngsolve PRIVATE cxx_std_17)

    # Use NGSolve TaskManager for parallel_for/parallel_reduce_sum
    target_compile_definitions(sparsesolv_ngsolve PRIVATE SPARSESOLV_USE_NGSOLVE_TASKMANAGER)

    # Install the Python module
    if(SKBUILD)
        # scikit-build-core: wheel-relative install path
        install(TARGETS sparsesolv_ngsolve
            LIBRARY DESTINATION sparsesolv_ngsolve
            RUNTIME DESTINATION sparsesolv_ngsolve)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
             "from .sparsesolv_ngsolve import *\n")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
                DESTINATION sparsesolv_ngsolve)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/sparsesolv_ngsolve.pyi"
                DESTINATION sparsesolv_ngsolve RENAME __init__.pyi)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/py.typed"
                DESTINATION sparsesolv_ngsolve)
    else()
        # Manual build: install next to NGSolve
        install(TARGETS sparsesolv_ngsolve
            LIBRARY DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve
            RUNTIME DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve)
    endif()
endif()

# ==============================================================================
# Header-only library installation
# ==============================================================================
include(GNUInstallDirs)
install(TARGETS sparsesolv
    EXPORT SparseSolvTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/sparsesolv
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT SparseSolvTargets
    FILE SparseSolvTargets.cmake
    NAMESPACE SparseSolv::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SparseSolv
)
