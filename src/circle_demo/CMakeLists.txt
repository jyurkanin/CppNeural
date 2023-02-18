find_package(SDL2 REQUIRED NO_MODULE)

add_executable(circle
  src/main.cpp
  src/MainWindow.cpp
  ../cpp_neural_lib/DenseLayer.cpp
  ../cpp_neural_lib/BaseLayer.cpp
  ../cpp_neural_lib/helpers.cpp
  ../cpp_neural_lib/types/Scalars.h
  )
target_link_libraries(circle ${SDL2_LIBRARIES})
