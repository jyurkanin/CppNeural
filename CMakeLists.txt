cmake_minimum_required (VERSION 2.8.11)
project(CppNeural)

find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -Ofast -mtune=native -march=native -DNDEBUG")
set(CMAKE_CXX_STANDARD 17)

include_directories(include /usr/include/eigen3)

add_library(cpp_neural
  src/helpers.cpp
  src/FinalLayer.cpp
  src/BaseLayer.cpp
  src/ConvLayer.cpp
  src/DenseLayer.cpp
  src/RecurrentLayer.cpp
  )


include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
  )
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories("src/")
add_executable(
  runUnitTests
  test/test_NormLayer.cpp
  test/test_ConvLayer.cpp

  src/helpers.cpp
  src/FinalLayer.cpp
  src/DenseLayer.cpp
  src/NormLayer.cpp
  src/BaseLayer.cpp
  src/ConvLayer.cpp
  )

#target_compile_options(runUnitTests PUBLIC -w -O0 -Wuninitialized -g)
target_link_libraries(
  runUnitTests
  GTest::gtest_main
  ${GTK3_LIBRARIES}
  )

include(GoogleTest)

gtest_discover_tests(runUnitTests)
