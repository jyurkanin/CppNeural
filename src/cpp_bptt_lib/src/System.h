#include <cpp_neural.h>
#include <Eigen/Core>



#pragma once

// I think this can be extended to automatically include derivatives
// The state can be augmented to include the necessary gradients?


namespace cpp_bptt
{

template<typename Scalar>
class System
{
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
  System(int state_dim, int control_dim) : m_state_dim(state_dim), m_control_dim(control_dim) {}
  ~System() {}

  virtual void getDefaultParams(VectorS &params) { params = VectorS::Zero(m_num_params); }
  virtual void getDefaultInitialState(VectorS &state) {state = VectorS::Zero(m_state_dim);}
  virtual void setParams(const VectorS &params) = 0;
  virtual void getParams(VectorS &params) = 0;
  virtual void forward(const VectorS &X, VectorS &Xd) = 0;
  virtual Scalar loss(const VectorS &gt_vec, VectorS &vec) = 0;
  
  void unflatten(const VectorF &vec, MatrixF &mat)
  {
    for(int i = 0; i < mat.rows(); i++)
    {
      int offset = i*mat.cols();
      for(int j = 0; j < mat.cols(); j++)
      {
	mat(i,j) = vec[offset+j];
      }
    }
  }  
  
  void setNumParams(int params)  { m_num_params = params; }
  void setNumSteps(int steps)    { m_num_steps = steps; }
  void setTimestep(double ts)     { m_timestep = ts; }
  void setLearningRate(double lr) { m_lr = lr; }

  int   getStateDim()     { return m_state_dim; }
  int   getControlDim()   { return m_control_dim; }
  int   getNumParams()    { return m_num_params; }
  int   getNumSteps()     { return m_num_steps; }
  double getTimestep()     { return m_timestep; }
  double getLearningRate() { return m_lr; }
  
private:
  int m_num_params;
  int m_state_dim;
  int m_control_dim;
  int m_num_steps;    // This is the number of states past x0
  double m_timestep;
  double m_lr;
};

}
  
