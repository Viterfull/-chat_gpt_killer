cmake_minimum_required(VERSION 3.2)

project(torchtest)

set(CMAKE_PREFIX_PATH /home/dimas/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${PROJECT_NAME} transformer.hpp transformer.cpp decoder_layer.cpp 
decoder_layer.hpp decoder.cpp decoder.hpp encoder_layer.cpp encoder_layer.hpp encoder.cpp encoder.hpp 
feed_forward.cpp feed_forward.hpp multi_head_attention.hpp multi_head_attention.cpp multi_head_cross_attention.hpp multi_head_cross_attention.cpp 
positional_enc.hpp positional_enc.cpp sentence_emb.hpp sentence_emb.cpp sequential_decoder.hpp sequential_decoder.cpp 
sequential_encoder.hpp sequential_encoder.cpp utils.hpp utils.cpp transformer.hpp transformer.cpp )
# add_executable(${PROJECT_NAME} test.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)  

