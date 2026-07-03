# NDVM build (Phase 1+ stub). See ../docs/NDVM_native_differentiable_vm.md section 18.
# CPU-first by design; CUDA and pybind11 are opt-in and added in later phases.
cmake_minimum_required(VERSION 3.18)
project(ndvm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall -Wextra)

option(NDVM_ENABLE_CUDA "Build the GPU backend (Phase 6)" OFF)
option(NDVM_ENABLE_PYTHON "Build the pybind11 Python extension (Phase 2)" OFF)

# Core native runtime. Phase 1 adds the front-end (sexpr) + native forward VM (interp).
# Phase 5 adds the multicore scheduler (parallel.cpp), which needs a threads library.
find_package(Threads REQUIRED)
file(GLOB NDVM_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(ndvm STATIC ${NDVM_SOURCES})
target_include_directories(ndvm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ndvm PUBLIC Threads::Threads)

# Phase-1 CLI driver used by the equivalence harness (tests/test_equivalence.py).
add_executable(ndvm_run tools/ndvm_run.cpp)
target_link_libraries(ndvm_run PRIVATE ndvm)

# Phase-5 parallel driver: a population of candidate evaluations across worker threads (determinism +
# scaling harness). NDVM_THREADS / NDVM_PAR_N / NDVM_PAR_DUMP / NDVM_B.
add_executable(ndvm_par tools/ndvm_par.cpp)
target_link_libraries(ndvm_par PRIVATE ndvm)

if(NDVM_ENABLE_CUDA)
  enable_language(CUDA)
  # Phase 6: persistent forward/backward kernels (see design section 12).
endif()

if(NDVM_ENABLE_PYTHON)
  # Phase 2: pybind11 extension exposing compile_evaluator/forward/backward (design section 14).
  # find_package(pybind11 REQUIRED); pybind11_add_module(...)
endif()

# Native unit tests (added in Phase 1+).
enable_testing()
