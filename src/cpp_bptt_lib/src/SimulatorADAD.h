#pragma once

#include <cpp_neural.h>
#include "Simulator.h"

namespace cpp_bptt
{

class SimulatorADAD : public Simulator<ADAD>
{
public:
  SimulatorADAD(std::shared_ptr<System<ADAD>> sys);
  ~SimulatorADAD();

  virtual void forward_backward(const VectorF &x0,
				const std::vector<VectorF> &gt_list,
				VectorF &gradient,
				double &loss);
};


}
