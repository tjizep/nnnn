cmake_minimum_required(VERSION 3.21)
project(nnnn)
message(STATUS "Project dir : ${CMAKE_SOURCE_DIR} " )
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/repo/Eigen3")
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/repo/Eigen3/cmake")
message(STATUS "Eigen include dir : ${EIGEN3_INCLUDE_DIR} " )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -std=c++17" )
find_package( Eigen3 REQUIRED )
include_directories(include ${EIGEN3_INCLUDE_DIR})


add_executable(nnnn train_network.cpp include/activations.h include/optimizers.h include/sparsity.h include/sse2neon.h include/model.h include/abstract_layer.h)
