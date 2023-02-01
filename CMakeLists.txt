cmake_minimum_required(VERSION 3.21)
project(nnnn)

set(EIGEN3_INCLUDE_DIR "C:\\Eigen3")
set(CMAKE_MODULE_PATH "C:\\Eigen3\\cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
find_package( Eigen3 REQUIRED )
include_directories(include ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen include dir : ${EIGEN3_INCLUDE_DIR} the eigen" )

add_executable(nnnn train_network.cpp include/activations.h include/optimizers.h include/sparsity.h include/sse2neon.h)
