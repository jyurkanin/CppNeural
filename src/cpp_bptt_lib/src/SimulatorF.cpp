#include "SimulatorF.h"

SimulatorF::SimulatorF(std::shared_ptr<System<float>> sys) : Simulator<float>(sys)
{
  
}

SimulatorF::~SimulatorF()
{
  
}

void SimulatorF::forward_backward(const VectorF &x0,
				  const std::vector<VectorF> &gt_list,
				  VectorF &gradient,
				  float &loss)
{
  VectorF gradient_acc         = VectorF::Zero(m_num_params);
  VectorF loss_x1_partial      = VectorF::Zero(m_state_dim);
  
  MatrixF x1_x_partial_mat     = MatrixF::Zero(m_state_dim, m_state_dim);
  VectorF x1_x_partial_vec     = VectorF::Zero(m_state_dim * m_state_dim);
  
  VectorF loss_theta_gradient  = VectorF::Zero(m_num_params);

  MatrixF x1_theta_partial_mat = MatrixF::Zero(m_state_dim, m_num_params);
  VectorF x1_theta_partial_vec = VectorF::Zero(m_state_dim * m_num_params);
  
  MatrixF x0_theta_jacobian    = MatrixF::Zero(m_state_dim, m_num_params);  //prev
  MatrixF x1_theta_jacobian    = MatrixF::Zero(m_state_dim, m_num_params);  //curr
  
  VectorF xk0 = x0;
  VectorF xk1(m_state_dim);
  
  float total_loss = 0;
  float loss_k;
  VectorF y0(1);
  y0[0] = 1;
  
  m_system->getParams(m_params);
  
  for(int i = 0; i < m_num_steps; i++)
  {
    // Advance state, compute partial_state_state, compute partial_state_param
    computePartials(xk0, m_params, xk1, x1_x_partial_vec, x1_theta_partial_vec);
    m_system->unflatten(x1_x_partial_vec, x1_x_partial_mat);
    m_system->unflatten(x1_theta_partial_vec, x1_theta_partial_mat);
        
    // Compute next state param jacobian
    x1_theta_jacobian = (x1_x_partial_mat * x0_theta_jacobian) + x1_theta_partial_mat;
    
    computePartialLossState(gt_list[i], xk1, loss_x1_partial, loss_k);
    total_loss += loss_k;
    
    // Compute Loss param gradient
    loss_theta_gradient = loss_x1_partial.transpose() * x1_theta_jacobian;
    gradient_acc += loss_theta_gradient;
    
    xk0 = xk1;
    x0_theta_jacobian = x1_theta_jacobian;
  }

  gradient = gradient_acc;
  loss = total_loss;
}

/*
void SimulatorF::computePartialLossState(const VectorF &gt_x1, const VectorF &x1, VectorF &loss_x1_partial)
{
  
}

void SimulatorF::computePartials(const VectorF &xk0,
				 const VectorF &theta,
				 VectorF &xk1,
				 VectorF &partial_state_state,
				 VectorF &partial_state_param)
{
  
}
*/
