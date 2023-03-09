#pragma once

#include <cpp_neural.h>
#include "Simulator.h"

class SimulatorADAD : public Simulator<ADAD>
{
public:
  SimulatorADAD(std::shared_ptr<System<ADAD>> sys);
  ~SimulatorADAD();

  virtual void forward_backward(const VectorF &x0,
				const std::vector<VectorF> &gt_list,
				VectorF &gradient,
				float &loss);
private:
  void computePartialLossState(const VectorF &gt_x1, const VectorF &x1, VectorF &loss_x1_partial);
  void computePartialStatePrevState(const VectorF &xk, const VectorF &theta, MatrixF &x1_x_partial, VectorF &xk1);
  void computePartialStateParams(const VectorF &x, const VectorF &theta, MatrixF &x1_theta_partial);
};
