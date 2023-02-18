cmake_minimum_required (VERSION 2.8.11)
project(CppNeural)

find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -Ofast -mtune=native -march=native -DNDEBUG")
set(CMAKE_CXX_STANDARD 17)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(src /usr/include/eigen3)
include(GNUInstallDirs)

add_subdirectory(src)
add_subdirectory(test)
