add_library(cpp_bptt
	src/solver.hpp
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
