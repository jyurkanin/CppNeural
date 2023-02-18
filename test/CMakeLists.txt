include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
  )
FetchContent_MakeAvailable(googletest)

enable_testing()


add_executable(
  runUnitTests
  test_NormLayer.cpp
  test_ConvLayer.cpp
  
  ../src/cpp_neural_lib/helpers.cpp
  ../src/cpp_neural_lib/FinalLayer.cpp
  ../src/cpp_neural_lib/DenseLayer.cpp
  ../src/cpp_neural_lib/NormLayer.cpp
  ../src/cpp_neural_lib/BaseLayer.cpp
  ../src/cpp_neural_lib/ConvLayer.cpp
  ../src/cpp_neural_lib/RecurrentLayer.cpp
  )

target_link_libraries(
  runUnitTests
  GTest::gtest_main
  )

include(GoogleTest)

gtest_discover_tests(runUnitTests)
