#include <memory>
#include <iostream>

#include <cpp_bptt.h>

#include <Generator.h>
#include <SimulatorADAD.h>
#include <types/Scalars.h>
#include <types/Tensors.h>

#include "CircleSystem.h"

int main()
{
  std::shared_ptr<CircleSystem<ADAD>> system = std::make_shared<CircleSystem<ADAD>>();
  std::shared_ptr<cpp_bptt::SimulatorADAD> simulator = std::make_shared<cpp_bptt::SimulatorADAD>(system);
  
  std::cout << "Num Params " << system->getNumParams() << "\n";
  
  cpp_bptt::Generator generator;
  generator.setSimulator(simulator);
  generator.initialize();  
}
