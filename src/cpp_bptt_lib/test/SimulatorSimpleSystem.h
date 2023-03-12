#pragma once

#include <cpp_neural.h>
#include "SimulatorF.h"

namespace cpp_bptt
{

class SimulatorSimpleSystem : public SimulatorF
{
public:
  SimulatorSimpleSystem(std::shared_ptr<System<float>> sys);
  ~SimulatorSimpleSystem();
    
  void computePartials(const VectorF &xk0,
		       const VectorF &theta,
		       const VectorF &xk1_gt,
		       VectorF &xk1,
		       VectorF &partial_state_state,
		       VectorF &partial_state_param,
		       VectorF &partial_loss_params,
		       VectorF &partial_loss_state,
		       float   &loss);

};

}
