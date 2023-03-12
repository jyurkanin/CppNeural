#include <cpp_neural.h>
#include "Simulator.h"

namespace cpp_bptt
{

class SimulatorF : public Simulator<float>
{
public:
  SimulatorF(std::shared_ptr<System<float>> sys);
  ~SimulatorF();

  virtual void forward_backward(const VectorF &x0,
				const std::vector<VectorF> &gt_list,
				VectorF &gradient,
				float &loss);
    
  virtual void computePartials(const VectorF &xk0,
			       const VectorF &theta,
			       const VectorF &xk1_gt,
			       VectorF &xk1,
			       VectorF &partial_state_state,
			       VectorF &partial_state_param,
			       VectorF &partial_loss_params,
			       VectorF &partial_loss_state,
			       float   &loss) = 0;
};

}


  
