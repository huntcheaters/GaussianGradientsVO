cmake_minimum_required(VERSION 2.4.6)
project(dvo_weight)

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# look for SuiteSparse, QGLViewer, and Eigen3.
#LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(
${PROJECT_SOURCE_DIR}/party3rd/tbb/include
${PROJECT_SOURCE_DIR}/include/dvo_weight)

link_directories(
${PROJECT_SOURCE_DIR}/party3rd/tbb/lib/ia32/vc12
${Boost_INCLUDE_DIR}/../../lib)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-msse2 -msse3")

# Retrieve all cpp files.
# Usage of GLOB means you need to run cmake every time a file is added/removed.
File(GLOB_RECURSE src_core src/core/*.cpp)
set(src_files ${src_core}  src/util/histogram.cpp)

# Compile to shared library.
#add_library(${PROJECT_NAME} SHARED ${src_files} )
add_library(${PROJECT_NAME} ${src_files})
#add_executable(${PROJECT_NAME} ${src_files})
#target_include_directories(dvo_core PUBLIC include)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} tbb.lib)

