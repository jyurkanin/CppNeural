find_package(SDL2 REQUIRED NO_MODULE)

include_directories(../cpp_neural_lib/)
include_directories(../cpp_bptt_lib/src)

add_executable(circle
  src/main.cpp
  src/MainWindow.cpp
  ../cpp_neural_lib/src/DenseLayer.cpp
  ../cpp_neural_lib/src/BaseLayer.cpp
  ../cpp_neural_lib/src/helpers.cpp
  ../cpp_neural_lib/src/types/Scalars.h
  )
target_link_libraries(circle ${SDL2_LIBRARIES})
