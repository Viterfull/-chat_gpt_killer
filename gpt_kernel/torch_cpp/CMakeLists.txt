cmake_minimum_required(VERSION 3.2)

project(torchtest)

set(CMAKE_PREFIX_PATH /home/dimas/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${PROJECT_NAME} utils.hpp utils.cpp feed_forward.hpp feed_forward.cpp positional_enc.hpp positional_enc.cpp multi_head_attention.hpp multi_head_attention.cpp)
# add_executable(${PROJECT_NAME} test.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)  

