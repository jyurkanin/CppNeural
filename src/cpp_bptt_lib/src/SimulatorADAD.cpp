#include "SimulatorADAD.h"

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

void SimulatorADAD::computePartialLossState(const VectorF &gt_x1, const VectorF &x1, VectorF &loss_x1_partial)
{
  
}

void SimulatorADAD::computePartialStatePrevState(const VectorF &xk, const VectorF &theta, MatrixF &x1_x_partial, VectorF &xk1)
{

}

void SimulatorADAD::computePartialStateParams(const VectorF &x, const VectorF &theta, MatrixF &x1_theta_partial)
{

}
