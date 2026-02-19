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
    find_package(NGSolve CONFIG REQUIRED
        HINTS "${NGSOLVE_INSTALL_DIR}"
    )
    message(STATUS "Found NGSolve: ${NGSOLVE_INCLUDE_DIRS}")

    add_ngsolve_python_module(sparsesolv_ngsolve ngsolve/python_module.cpp)
    target_include_directories(sparsesolv_ngsolve PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(sparsesolv_ngsolve PRIVATE cxx_std_17)

    # Use NGSolve TaskManager for parallel_for/parallel_reduce_sum
    target_compile_definitions(sparsesolv_ngsolve PRIVATE SPARSESOLV_USE_NGSOLVE_TASKMANAGER)

    # Install the Python module next to NGSolve
    install(TARGETS sparsesolv_ngsolve
        LIBRARY DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve
        RUNTIME DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve
    )
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
