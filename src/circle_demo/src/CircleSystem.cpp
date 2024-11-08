#include "CircleSystem.h"


template<typename Scalar>
CircleSystem<Scalar>::CircleSystem() : cpp_bptt::System<Scalar>(2,1), m_network(2,4,2)
{
  this->setNumParams(m_network.getNumParams());
  this->setNumSteps(4000);
  this->setTimestep(0.1);
  this->setLearningRate(1e-2f);
}

template<typename Scalar>
CircleSystem<Scalar>::~CircleSystem()
{
  
}

template<typename Scalar>
void CircleSystem<Scalar>::setParams(const VectorS &params)
{
  int idx = 0;
  m_network.setParams(params, idx);
}

template<typename Scalar>
void CircleSystem<Scalar>::getParams(VectorS &params)
{
  int idx = 0;
  m_network.getParams(params, idx);
}


template<typename Scalar>
void CircleSystem<Scalar>::forward(const VectorS &X, VectorS &Xd)
{
  MatrixS X_mat(X.rows(),1);
  MatrixS Xd_mat(Xd.rows(),1);
  
  for(int i = 0; i < X.rows(); i++)
  {
    X_mat(i,0) = X[i];
  }
  
  m_network.process(Xd_mat, X_mat);

  for(int i = 0; i < Xd.rows(); i++)
  {
    Xd[i] =  Xd_mat(i,0);
  }
}

template<typename Scalar>
Scalar CircleSystem<Scalar>::loss(const VectorS &gt_vec, VectorS &vec)
{
  VectorS d_vec(this->getStateDim());
  
  Scalar sum_err(0.0);
  
  Scalar radius = vec[0]*vec[0] + vec[1]*vec[1];
  Scalar err = m_target_radius - radius;
  sum_err = err*err;
  
  forward(vec, d_vec);
  
  Scalar norm = CppAD::exp(-10*(CppAD::abs(d_vec[0]) + CppAD::abs(d_vec[1])));
  sum_err += norm;
  
  return sum_err;
  
}



template class CircleSystem<double>;
template class CircleSystem<ADF>;
template class CircleSystem<ADAD>;
