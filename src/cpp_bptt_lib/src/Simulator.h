#pragma once
#include "System.h"

#include <memory>


/*
 * There will be two versions of this class. SimulatorAD, SimulatorCodeGen
 * The goal for this class is to do forward and backwards passes.
 * To simulate forward, then backpropagate to train.
 */

template<typename Scalar>
class Simulator
{
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
  Simulator(std::shared_ptr<System<Scalar>> sys) : m_system(sys)
  {
    m_timestep = sys->getTimestep();
    m_num_steps = sys->getNumSteps();
    m_state_dim = sys->getStateDim();
    m_control_dim = sys->getControlDim();
    m_num_params = sys->getNumParams();
    
    m_params = VectorF::Ones(m_num_params);
  }
  ~Simulator(){}

  void forward(const VectorS &x0, std::vector<VectorS> &x_list)
  {
    assert(x0.size() == m_system->getStateDim());
    
    VectorAD xk = x0;
    VectorAD xk1(m_system->getStateDim());
    
    for(int i = 0; i < m_system->getNumSteps(); i++)
    { 
      integrate(xk, xk1);
      xk = xk1;
      x_list[i] = xk;
    }
  }
  
  virtual void forward_backward(const VectorF &x0,
				const std::vector<VectorF> &gt_list,
				VectorF &gradient,
				float &loss) = 0;
  
  void train(const VectorF &x0, const std::vector<VectorF> &gt_list)
  {
    VectorF gradient;
    float loss;
    
    forward_backward(x0, gt_list, gradient, loss);
    m_params -= gradient * m_system->getLearningRate();
  }
  
  /// @param theta params
  /// @param Xk This is a tensor of shape (num_samples, state_dim)
  /// @param Xk1 same as Xk but the next discrete time step
  void integrate(const VectorS &Xk, VectorS &Xk1)
  {
    rk4(Xk, Xk1);
  }
  
  std::shared_ptr<System<Scalar>> getSystem() { return m_system; }
  void setParams(const VectorS &params) { m_system->setParams(params); }
protected:
  float m_timestep;
  int   m_num_steps;
  int   m_control_dim;
  int   m_state_dim;
  int   m_num_params;
  VectorF m_params;
  std::shared_ptr<System<Scalar>> m_system;
  
private:
  void rk4(const VectorS &Xk, VectorS &Xk1)
  {
    Scalar ts(m_timestep);
    Scalar ts_6(m_timestep/6.0f);
    Scalar c1(1);
    Scalar c2(2);
    Scalar c3(2);
    Scalar c4(1);
    
    VectorS k1(m_state_dim);
    VectorS k2(m_state_dim);
    VectorS k3(m_state_dim);
    VectorS k4(m_state_dim);
    VectorS temp(m_state_dim);
    
    m_system->forward(Xk, k1);
    temp = Xk + .5*ts*k1;

    m_system->forward(temp, k2);
    temp = Xk + .5*ts*k2;
    
    m_system->forward(temp, k3);
    temp = Xk + ts*k3;

    m_system->forward(temp, k4);
    Xk1 = Xk + ((ts_6)*(k1 + c2*k2 + c3*k3 + k4));
  }

};
