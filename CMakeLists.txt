cmake_minimum_required(VERSION 3.19)
project(nnnn)
find_package(Threads)
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/repo/eigen")
set(JSON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/repo/json/include")
set(SSE2NEON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/repo/sse2neon")
#set(RVA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/repo/recursive-variant/include")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/repo/json/cmake")
message(STATUS "Eigen include dir : ${EIGEN3_INCLUDE_DIR} " )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -msse4.1 -std=c++20" )


include_directories(include ${EIGEN3_INCLUDE_DIR} ${JSON_INCLUDE_DIR} ${SSE2NEON_INCLUDE_DIR} ${RVA_INCLUDE_DIR})


add_executable(nnnn train_network.cpp include/activations.h include/optimizers.h include/sparsity.h repo/sse2neon/sse2neon.h include/model.h include/abstract_layer.h include/model_json.h include/ensemble.h include/rnn.h include/diffrentiable.h include/graph.h include/message.h)
target_link_libraries(nnnn ${CMAKE_THREAD_LIBS_INIT})