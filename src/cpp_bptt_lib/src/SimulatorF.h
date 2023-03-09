#include <cpp_neural.h>
#include "Simulator.h"

class SimulatorF : public Simulator<float>
{
public:
  SimulatorF(std::shared_ptr<System<float>> sys);
  ~SimulatorF();

  virtual void forward_backward(const VectorF &x0,
				const std::vector<VectorF> &gt_list,
				VectorF &gradient,
				float &loss);
  
  virtual void computePartialLossState(const VectorF &gt_x1,
				       const VectorF &x1,
				       VectorF &loss_x1_partial,
				       float &loss) = 0;
  
  virtual void computePartials(const VectorF &xk0,
			       const VectorF &theta,
			       VectorF &xk1,
			       VectorF &partial_state_state,
			       VectorF &partial_state_param) = 0;
};
