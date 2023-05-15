cmake_minimum_required(VERSION 3.2)

project(torchtest)

set(CMAKE_PREFIX_PATH /home/dimas/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_library(${PROJECT_NAME} transformer_cpp.hpp transformer_cpp.cpp)
add_executable(${PROJECT_NAME} test.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)  

