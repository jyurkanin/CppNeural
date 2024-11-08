#pragma once

#include <cpp_neural.h>
#include "Simulator.h"

namespace cpp_bptt
{

class SimulatorF : public Simulator<double>
{
public:
  SimulatorF(std::shared_ptr<System<double>> sys);
  ~SimulatorF();

  virtual void forward_backward(const VectorF &x0,
				const std::vector<VectorF> &gt_list,
				VectorF &gradient,
				double &loss);
    
  virtual void computePartials(const VectorF &xk0,
			       const VectorF &theta,
			       const VectorF &xk1_gt,
			       VectorF &xk1,
			       VectorF &partial_state_state,
			       VectorF &partial_state_param,
			       VectorF &partial_loss_params,
			       VectorF &partial_loss_state,
			       double   &loss) = 0;
};

}


  
