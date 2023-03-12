#include "SimulatorAD.h"

#include <assert.h>

namespace cpp_bptt
{

void printMat(const MatrixF &mat)
{
  for(int i = 0; i < mat.rows(); i++)
  {
    for(int j = 0; j < mat.cols()-1; j++)
    {
      std::cout << mat(i,j) << ",";
    }
    std::cout << mat(i, mat.cols()-1) << "\n";
  }
}

SimulatorAD::SimulatorAD(std::shared_ptr<System<ADF>> sys) : Simulator<ADF>(sys)
{
  precomputePartialLossParams();
  precomputePartialLossState();
  precomputePartialStatePrevState();
  precomputePartialStateParams();
  
  clearGradients();
}

SimulatorAD::~SimulatorAD(){}



void SimulatorAD::clearGradients()
{
  
}

// I think this is correct
void SimulatorAD::forward_backward(const VectorF &x0,
				   const std::vector<VectorF> &gt_list,
				   VectorF &gradient,
				   float &total_loss)
{
  VectorF gradient_acc         = VectorF::Zero(m_num_params);
  VectorF loss_x1_partial      = VectorF::Zero(m_state_dim);
  VectorF loss_state_dynamic_vars = VectorF::Zero(m_state_dim + m_num_params);
  VectorF loss_params_dynamic_vars = VectorF::Zero(m_state_dim*2);
  VectorF loss_params_partial  = VectorF::Zero(m_num_params);
  MatrixF x1_x_partial_mat     = MatrixF::Zero(m_state_dim, m_state_dim);
  VectorF x1_x_partial_vec     = VectorF::Zero(m_state_dim * m_state_dim);
  VectorF loss_theta_gradient  = VectorF::Zero(m_num_params);
  MatrixF x1_theta_partial_mat = MatrixF::Zero(m_state_dim, m_num_params);
  VectorF x1_theta_partial_vec = VectorF::Zero(m_state_dim * m_num_params);
  MatrixF x_theta_jacobian     = MatrixF::Zero(m_state_dim, m_num_params);  //prev
  MatrixF x1_theta_jacobian    = MatrixF::Zero(m_state_dim, m_num_params);  //curr
  VectorF xk = x0;
  VectorF xk1;

  //boiler plate
  total_loss = 0;
  VectorF loss(1);
  VectorF y0(1);
  y0[0] = 1;

  VectorAD params(m_system->getNumParams());
  m_system->getParams(params);
  for(int i = 0; i < m_system->getNumParams(); i++)
  {
    m_params[i] = CppAD::Value(params[i]);
  }
  
  m_partial_state_prev_state.new_dynamic(m_params);
  for(int i = 0; i < m_num_steps; i++)
  {
    // Advance state
    xk1 = m_partial_state_prev_state.Forward(0, xk);
    
    // Compute state state jacobian
    x1_x_partial_vec = m_partial_state_prev_state.Jacobian(xk);    
    m_system->unflatten(x1_x_partial_vec, x1_x_partial_mat);
        
    // Compute state param partial
    m_partial_state_params.new_dynamic(xk);
    x1_theta_partial_vec = m_partial_state_params.Jacobian(m_params);
    m_system->unflatten(x1_theta_partial_vec, x1_theta_partial_mat);
    
    // Compute next state param jacobian
    x1_theta_jacobian = (x1_x_partial_mat * x_theta_jacobian) + x1_theta_partial_mat;
    
    // Compute loss and loss state partials
    int idx = 0;
    for(int j = 0; j < m_params.size(); j++)
    {
      loss_state_dynamic_vars[idx+j] = m_params[j];
    }
    idx += m_params.size();
    for(int j = 0; j < gt_list[i].size(); j++)
    {
      loss_state_dynamic_vars[idx+j] = gt_list[i][j];
    }
    idx += gt_list[i].size();

    m_partial_loss_state.new_dynamic(loss_state_dynamic_vars);
    loss = m_partial_loss_state.Forward(0, xk1);
    total_loss += loss[0];
    loss_x1_partial = m_partial_loss_state.Reverse(1, y0);     // Compute Loss state partial
    
    // Compute Loss params partial
    idx = 0;
    for(int j = 0; j < xk1.size(); j++)
    {
      loss_params_dynamic_vars[idx+j] = xk1[j];
    }
    idx += xk1.size();
    for(int j = 0; j < gt_list[i].size(); j++)
    {
      loss_params_dynamic_vars[idx+j] = gt_list[i][j];
    }
    idx += gt_list[i].size();
    
    m_partial_loss_params.new_dynamic(loss_params_dynamic_vars);
    m_partial_loss_params.Forward(0, m_params);
    loss_params_partial = m_partial_loss_params.Reverse(1, y0);
    
    // Compute Loss param gradient
    loss_theta_gradient = (loss_x1_partial.transpose() * x1_theta_jacobian).transpose() + loss_params_partial;
    
    // Acumulate gradient of the loss at each point in time
    gradient_acc += loss_theta_gradient;
    
    xk = xk1;
    x_theta_jacobian = x1_theta_jacobian;
  }

  gradient = gradient_acc;
}

void SimulatorAD::precomputePartialLossState()
{
  VectorAD x0(m_system->getStateDim());
  VectorAD gt_x0(m_system->getStateDim());
  VectorAD loss(1);
  VectorAD theta(m_system->getNumParams());
  VectorAD loss_state_dynamic_vars(m_state_dim + m_num_params);
  
  CppAD::Independent(x0, loss_state_dynamic_vars);
  int idx = 0;
  for(int j = 0; j < theta.size(); j++)
  {
    theta[j] = loss_state_dynamic_vars[idx+j];
  }
  idx += theta.size();
  for(int j = 0; j < gt_x0.size(); j++)
  {
    gt_x0[j] = loss_state_dynamic_vars[idx+j];
  }
  idx += gt_x0.size();

  m_system->setParams(theta);
  loss[0] = m_system->loss(gt_x0, x0);
  m_partial_loss_state = CppAD::ADFun<float>(x0,loss);
}

void SimulatorAD::precomputePartialLossParams()
{
  VectorAD x0(m_system->getStateDim());
  VectorAD gt_x0(m_system->getStateDim());
  VectorAD loss(1);
  VectorAD dynamic_vars(x0.size() + gt_x0.size());
  VectorAD theta(m_system->getNumParams());
  
  CppAD::Independent(theta, dynamic_vars);
  int idx = 0;
  for(int i = 0; i < x0.size(); i++)
  {
    x0[i] = dynamic_vars[i+idx];
  }
  idx += x0.size();
  for(int i = 0; i < gt_x0.size(); i++)
  {
    gt_x0[i] = dynamic_vars[i+idx];
  }
  idx += gt_x0.size();

  m_system->setParams(theta);
  loss[0] = m_system->loss(gt_x0, x0);
  m_partial_loss_params = CppAD::ADFun<float>(theta,loss);
}

  
void SimulatorAD::precomputePartialStatePrevState()
{
  VectorAD x0(m_system->getStateDim());
  VectorAD x1(m_system->getStateDim());
  VectorAD params(m_system->getNumParams());
  
  CppAD::Independent(x0, params);

  m_system->setParams(params);
  integrate(x0, x1);  
  m_partial_state_prev_state = CppAD::ADFun<float>(x0,x1);
}

void SimulatorAD::precomputePartialStateParams()
{
  VectorAD x0(m_system->getStateDim());
  VectorAD x1(m_system->getStateDim());
  VectorAD params(m_system->getNumParams());
  
  CppAD::Independent(params, x0);
  
  m_system->setParams(params);
  integrate(x0, x1);
  m_partial_state_params = CppAD::ADFun<float>(params,x1);
}

}
