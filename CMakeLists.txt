cmake_minimum_required (VERSION 2.8.11)
project(CppNeural)

find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -Ofast -mtune=native -march=native -DNDEBUG")
set(CMAKE_CXX_STANDARD 17)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(/usr/include/eigen3)
include(GNUInstallDirs)

include_directories(src/cpp_neural_lib/src/)
include_directories(src/cpp_bptt_lib/src/)
include_directories(src/circle_demo/src/)


include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
  )
FetchContent_MakeAvailable(googletest)
enable_testing()


add_subdirectory(src)
