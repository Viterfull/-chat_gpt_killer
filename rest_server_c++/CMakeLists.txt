cmake_minimum_required(VERSION 3.10)

project("REST_API_FOR_TRANSLATER")

set(CMAKE_CXX_STANDARD 11)

find_package(cpprestsdk REQUIRED)
find_package(PythonLibs REQUIRED)

add_executable(server server.cpp)

target_link_libraries(server ${PYTHON_LIBRARIES} cpprestsdk::cpprest)
include_directories(${PYTHON_INCLUDE_DIRS})



# add_executable(my_cpp my_cpp.cpp)
# set_property(TARGET my_cpp PROPERTY CXX_STANDARD 20)



