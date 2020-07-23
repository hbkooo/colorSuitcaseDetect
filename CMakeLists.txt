cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

set(TENSORRT_ROOT /data/hbk/TensorRT-6.0.1.5/)

#add wrapper
set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/TrtNet)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

#build run_test
#add_executable(run_test run_test.cpp )
#target_link_libraries(run_test TrtNet ${OpenCV_LIBS})
#build main
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(main main.cpp
        suitcase_detector/suitcase_detector.cpp suitcase_detector/suitcase_detector.h
        suitcase_detector/suitcaseClassify.cpp suitcase_detector/suitcaseClassify.h)
target_link_libraries(main TrtNet ${OpenCV_LIBS})

install(TARGETS main DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
