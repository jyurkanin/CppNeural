#include "BaseLayer.h"

template<typename Scalar>
BaseLayer<Scalar>::BaseLayer(int num_in, int num_out)
{
  m_num_in = num_in;
  m_num_out = num_out;
}

template<typename Scalar>
BaseLayer<Scalar>::~BaseLayer(){}

template<typename Scalar>
void BaseLayer<Scalar>::reset(){}

template<typename Scalar>
int BaseLayer<Scalar>::getNumInputs(){return m_num_in;}

template<typename Scalar>
int BaseLayer<Scalar>::getNumOutputs(){return m_num_out;}

template class BaseLayer<ADF>;
template class BaseLayer<ADAD>;
