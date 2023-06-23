cmake_minimum_required (VERSION 3.13)
project(GMVO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
#set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED)

#message(STATUS "Path is: â€?{OpenCV_PATH}'")
message("\n=====OpenCV Library=====")
message(STATUS "[Directory]:    ${OpenCV_DIR}")
message(STATUS "[Version]:      ${OpenCV_VERSION}")
message(STATUS "[Libraries]:    ${OpenCV_LIBS}")
message(STATUS "[Include Path]: ${OpenCV_INCLUDE_DIRS}")
message( "========================\n")

# DVO weight added on 07/10/2022.
include_directories(${GMVO_SOURCE_DIR}/include
                    ${GMVO_SOURCE_DIR}/party3rd/Eigen
${GMVO_SOURCE_DIR}/include/dvo_weight
${GMVO_SOURCE_DIR}/src/dvo_weight/party3rd/tbb/include
)

link_directories(${GMVO_SOURCE_DIR}/src/dvo_weight/party3rd/tbb/lib/ia32/vc12
)

file(GLOB SOURCES "include/*.h" "src/*.cpp")

# DVO Weight # Added on 07/10/2022.
add_subdirectory(${GMVO_SOURCE_DIR}/src/dvo_weight)

add_executable(GMVO ${SOURCES})

# DVO weight added as an independent library. 07/10/2022.
target_link_libraries (GMVO  ${OpenCV_LIBS}  dvo_weight)
   
  