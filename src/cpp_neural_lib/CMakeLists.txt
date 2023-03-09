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
  src/RecurrentLayer.cpp
  src/CSVHandler.cpp
  src/types/Scalars.h
  src/types/Tensors.h
  )

target_link_libraries(
  runUnitTests
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(runUnitTests)



add_library(cpp_neural
  src/helpers.cpp
  src/FinalLayer.cpp
  src/DenseLayer.cpp
  src/NormLayer.cpp
  src/BaseLayer.cpp
  src/ConvLayer.cpp
  src/RecurrentLayer.cpp
  src/CSVHandler.cpp
  src/types/Scalars.h
  src/types/Tensors.h
  )

install(
	TARGETS cpp_neural
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpp_neural
	FILES_MATCHING PATTERN "*.h"
)
