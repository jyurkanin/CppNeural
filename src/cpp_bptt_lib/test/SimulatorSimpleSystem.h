#pragma once

#include <cpp_neural.h>
#include "SimulatorF.h"


class SimulatorSimpleSystem : public SimulatorF
{
public:
  SimulatorSimpleSystem(std::shared_ptr<System<float>> sys);
  ~SimulatorSimpleSystem();
  
  void computePartialLossState(const VectorF &gt_x1,
			       const VectorF &x1,
			       VectorF &loss_x1_partial,
			       float &loss);
  
  void computePartials(const VectorF &xk0,
		       const VectorF &theta,
		       VectorF &xk1,
		       VectorF &partial_state_state,
		       VectorF &partial_state_param);
};
