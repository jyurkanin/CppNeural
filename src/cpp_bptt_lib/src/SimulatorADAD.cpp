#include "SimulatorADAD.h"

namespace cpp_bptt
{

SimulatorADAD::SimulatorADAD(std::shared_ptr<System<ADAD>> sys) : Simulator<ADAD>(sys)
{
  m_timestep = sys->getTimestep();
  m_num_steps = sys->getNumSteps();
  m_state_dim = sys->getStateDim();
  m_control_dim = sys->getControlDim();
  m_num_params = sys->getNumParams();
  
  m_params = VectorF::Ones(m_num_params);  
}

SimulatorADAD::~SimulatorADAD()
{
  
}

void SimulatorADAD::forward_backward(const VectorF &x0,
					const std::vector<VectorF> &gt_list,
					VectorF &gradient,
					float &loss)
{
  
}

}
