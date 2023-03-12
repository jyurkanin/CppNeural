#include "DenseLayer.h"
#include <assert.h>

template<typename Scalar>
DenseLayer<Scalar>::DenseLayer(int num_in, int num_out) : BaseLayer<Scalar>(num_in, num_out)
{
  m_num_in = num_in;
  m_num_out = num_out;
  
  Scalar small(.1);
  
  m_weights = small*Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(num_out, num_in);
  m_biases = small*Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(num_out);

  for(int i = 0; i < m_weights.rows(); i++)
  {
    for(int j = 0; j < m_weights.cols(); j++)
    {
      m_weights(i,j) = m_weights(i,j) - (.5f*small);
    }

    //m_biases[i] = m_biases[i] - (.5f*small); //prevent dead relus?
  }
}

template<typename Scalar>
DenseLayer<Scalar>::~DenseLayer()
{

}

template<typename Scalar>
void DenseLayer<Scalar>::process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &output, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input)
{
  assert(output.rows() == m_num_out);
  assert(input.rows() == m_num_in);

  output = m_weights*input;
  for(int i = 0; i < output.rows(); i++)
  {
    for(int j = 0; j < output.cols(); j++)
    {
      output(i,j) = output(i,j) + m_biases[i];
    }
  }
}

template<typename Scalar>
void DenseLayer<Scalar>::setParams(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
{
  for(int i = 0; i < m_weights.rows(); i++)
  {
    for(int j = 0; j < m_weights.cols(); j++)
    {
      m_weights(i,j) = params[idx];
      idx++;
    }
  }

  for(int i = 0; i < m_biases.rows(); i++)
  {
    m_biases[i] = params[idx];
    idx++;
  }

}

template<typename Scalar>
unsigned DenseLayer<Scalar>::getNumParams()
{
  return (m_weights.rows()*m_weights.cols()) + m_biases.rows();
}


template<typename Scalar>
void DenseLayer<Scalar>::getParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
{
  for(int i = 0; i < m_weights.rows(); i++)
    {
      for(int j = 0; j < m_weights.cols(); j++)
        {
          params[idx] = m_weights(i,j);
          idx++;
        }
    }

  for(int i = 0; i < m_biases.rows(); i++)
    {
      params[idx] = m_biases[i];
      idx++;
    }
}

template<typename Scalar>
void DenseLayer<Scalar>::zeroBias()
{
  Scalar zero(0);
  for(int i = 0; i < m_biases.size(); i++)
  {
    m_biases[i] = zero;
  }
}

template class DenseLayer<float>;
template class DenseLayer<ADF>;
template class DenseLayer<ADAD>;
