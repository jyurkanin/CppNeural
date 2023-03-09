#include "CircleSystem.h"



CircleSystem::CircleSystem() : cpp_bptt::System<Scalar>(2,1), m_network(2,4,2)
{
  setNumSteps(1000);
  setTimestep(0.01);
  setBatchSize(4);
  setNumParams(m_network.getNumParams());
}

CircleSystem::~CircleSystem()
{
  
}

void CircleSystem::setParams(const VectorS &params)
{
  int idx = 0;
  m_network.setParams(params, idx);
}


void CircleSystem::forward(const VectorS &X, VectorS &Xd)
{
  m_network.process(Xd, X);
}


Scalar CircleSystem::loss(const VectorS &gt_vec, VectorS &vec)
{
  Scalar sum_err = 0;
  for(int j = 0; j < gt_vec.cols(); j++)
  {
    Scalar err = gt_vec[j] - vec[j];
    sum_err += err*err;
  }
  return sum_err;
  
}
