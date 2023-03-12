find_package(SDL2 REQUIRED NO_MODULE)

include_directories(../cpp_neural_lib/src)
include_directories(../cpp_bptt_lib/src)

add_executable(circle
  src/main.cpp
  src/MainWindow.cpp
  src/CircleSystem.cpp
  src/CircleSimulator.cpp
  ../cpp_bptt_lib/src/SimulatorF.cpp
  ../cpp_neural_lib/src/DenseLayer.cpp
  ../cpp_neural_lib/src/BaseLayer.cpp
  ../cpp_neural_lib/src/helpers.cpp
  ../cpp_neural_lib/src/types/Scalars.h
  )
target_link_libraries(circle ${SDL2_LIBRARIES})

add_executable(gen_circle
	src/generate_circle.cpp
	src/CircleSystem.cpp
	../cpp_neural_lib/src/DenseLayer.cpp
	../cpp_neural_lib/src/BaseLayer.cpp
	../cpp_neural_lib/src/helpers.cpp
	../cpp_neural_lib/src/types/Scalars.h
	../cpp_bptt_lib/src/SimulatorADAD.cpp
	../cpp_bptt_lib/src/SimulatorF.cpp
	../cpp_bptt_lib/src/Generator.cpp
)
