add_library(cpp_neural
  helpers.cpp
  FinalLayer.cpp
  DenseLayer.cpp
  NormLayer.cpp
  BaseLayer.cpp
  ConvLayer.cpp
  RecurrentLayer.cpp
  CSVHandler.cpp
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
