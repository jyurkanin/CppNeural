#include "SimpleSystem.h"
#include <iostream>

namespace cpp_bptt
{

template<typename Scalar>
SimpleSystem<Scalar>::SimpleSystem() : m_params(2,2), System<Scalar>(2,1)
{
  this->setNumParams(4);
  this->setNumSteps(100);
  this->setTimestep(.1);
  this->setLearningRate(0.01);
}

template<typename Scalar>
SimpleSystem<Scalar>::~SimpleSystem()
{
  
}

template<typename Scalar>
void SimpleSystem<Scalar>::setParams(const VectorS &params)
{
  m_params(0,0) = params[0];
  m_params(0,1) = params[1];
  m_params(1,0) = params[2];
  m_params(1,1) = params[3];
}

template<typename Scalar>
void SimpleSystem<Scalar>::getParams(VectorS &params)
{
  params[0] = m_params(0,0);
  params[1] = m_params(0,1);
  params[2] = m_params(1,0);
  params[3] = m_params(1,1);
}

template<typename Scalar>
void SimpleSystem<Scalar>::forward(const VectorS &X, VectorS &Xd)
{
  Xd = (m_params*X);
}

template<typename Scalar>
Scalar SimpleSystem<Scalar>::loss(const VectorS &gt_vec, VectorS &vec)
{
  Scalar sum(0.0);
  for(int i = 0; i < gt_vec.size(); i++)
  {
    Scalar err = gt_vec[i] - vec[i];
    sum += err*err;
  }

  VectorS Xd(2);
  forward(vec, Xd);
  return sum + Xd[0] + Xd[1];
}


template class SimpleSystem<float>;
template class SimpleSystem<ADF>;
template class SimpleSystem<ADAD>;

} //cpp_bptt
