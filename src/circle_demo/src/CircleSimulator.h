#pragma once

#include <memory>

#include <cpp_bptt.h>
#include <SimulatorF.h>
#include <System.h>

class CircleSimulator : public cpp_bptt::SimulatorF
{
public:
  CircleSimulator(std::shared_ptr<cpp_bptt::System<double>> sys);
  ~CircleSimulator();

  void computePartialLossState(const VectorF &gt_x1,
			       const VectorF &x1,
			       VectorF &loss_x1_partial,
			       double &loss);

  void computePartials(const VectorF &xk0,
		       const VectorF &theta,
		       const VectorF &xk1_gt,
		       VectorF &xk1,
		       VectorF &partial_state_state,
		       VectorF &partial_state_param,
		       VectorF &partial_loss_params,
		       VectorF &partial_loss_state,
		       double   &loss);

};
