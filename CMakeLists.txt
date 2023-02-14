cmake_minimum_required(VERSION 3.21)
project(nnnn)
message(STATUS "Project dir : ${CMAKE_SOURCE_DIR} " )
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/repo/eigen")
set(JSON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/repo/json/include")
set(SSE2NEON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/repo/sse2neon")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/repo/eigen/cmake" "${CMAKE_SOURCE_DIR}/repo/json/cmake")
message(STATUS "Eigen include dir : ${EIGEN3_INCLUDE_DIR} " )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -std=c++17" )


include_directories(include ${EIGEN3_INCLUDE_DIR} ${JSON_INCLUDE_DIR} ${SSE2NEON_INCLUDE_DIR})


add_executable(nnnn train_network.cpp include/activations.h include/optimizers.h include/sparsity.h repo/sse2neon/sse2neon.h include/model.h include/abstract_layer.h include/model_json.h)
