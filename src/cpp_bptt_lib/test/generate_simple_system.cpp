#include <memory>
#include <fenv.h>

#include "Generator.h"
#include "SimpleSystem.h"
#include "SimulatorADAD.h"
#include <types/Scalars.h>
#include <types/Tensors.h>




int main(){
  // feenableexcept(FE_INVALID | FE_OVERFLOW);
  std::shared_ptr<SimpleSystem<ADAD>> system = std::make_shared<SimpleSystem<ADAD>>();
  std::shared_ptr<SimulatorADAD> simulator = std::make_shared<SimulatorADAD>(system);
  
  Generator generator;
  generator.setSimulator(simulator);
  generator.initialize();
}
