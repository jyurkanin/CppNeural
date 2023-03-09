#pragma once

#include <cpp_neural.h>
#include <types/Tensors.h>
#include <types/Scalars.h>
#include <vector>
#include "Simulator.h"

// This is going to implement the functionality associated with the CppAD type.
// It's going to autocompute the necessary gradients ahead of time.
// It's also going to store the accumulated gradients and do the chain rule for-loop magic for back prop through time
class SimulatorAD : public Simulator<ADF>
{
public:
  SimulatorAD(std::shared_ptr<System<ADF>> sys);
  ~SimulatorAD();

  
  void clearGradients();
  
  virtual void forward_backward(const VectorF &x0,
				const std::vector<VectorF> &gt_list,
				VectorF &gradient,
				float &loss);
  
private:
  void precomputePartialLossState();      // Partial of Loss wrt the current state
  void precomputePartialStatePrevState(); // Partial of state wrt the previous state
  void precomputePartialStateParams();    // Partial of state wrt the params
  
  CppAD::ADFun<float> m_partial_loss_state;
  CppAD::ADFun<float> m_partial_state_prev_state;
  CppAD::ADFun<float> m_partial_state_params;
};
