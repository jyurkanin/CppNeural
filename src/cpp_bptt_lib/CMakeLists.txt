#include_directories(../cpp_neural_lib/)



add_executable(
  runUnitTest_bptt
  test/SimpleSystem.cpp
  test/SimulatorSimpleSystem.cpp
  test/test_SimpleSystem.cpp
  
  src/Generator.cpp
  src/SimulatorF.cpp
  src/SimulatorAD.cpp
  
  ../cpp_neural_lib/src/types/Scalars.h
  ../cpp_neural_lib/src/types/Tensors.h
)

target_link_libraries(
  runUnitTest_bptt
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(runUnitTest_bptt)


add_executable(
  generate_simple_system
  src/Generator.cpp
  src/SimulatorADAD.cpp
  test/SimpleSystem.cpp
  test/generate_simple_system.cpp
)



add_library(cpp_bptt
	src/Generator.cpp
)

install(
	TARGETS cpp_bptt
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpp_bptt
	FILES_MATCHING PATTERN "*.h"
)
