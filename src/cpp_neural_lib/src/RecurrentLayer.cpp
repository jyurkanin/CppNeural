#include "RecurrentLayer.h"
#include <assert.h>

template<typename Scalar>
RecurrentLayer<Scalar>::RecurrentLayer(int num_in, int num_recurrent, int num_out) : BaseLayer<Scalar>(num_in, num_out)
{
  m_num_in = num_in;
  m_num_out = num_out;
  m_num_recurrent = num_recurrent;

  m_weights = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(num_out+num_recurrent, num_in+num_recurrent);
  m_biases = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(num_out+num_recurrent);

  for(int i = 0; i < m_weights.rows(); i++)
  {
    for(int j = 0; j < m_weights.cols(); j++)
    {
      m_weights(i,j) = m_weights(i,j) - .5f;
    }

    m_biases[i] = m_biases[i] - .5f;
  }
}

template<typename Scalar>
RecurrentLayer<Scalar>::~RecurrentLayer()
{

}

template<typename Scalar>
void RecurrentLayer<Scalar>::process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &output, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input)
{
  assert(output.rows() == m_num_out);
  assert(input.rows() == m_num_in);

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> input_aug(m_num_in + m_num_recurrent);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> output_aug(m_num_out + m_num_recurrent);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> prev_output(m_num_recurrent);
  prev_output = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(m_num_recurrent);

  for(int j = 0; j < output.cols(); j++)
  {
    for(int i = 0; i < m_num_in; i++)
    {
      input_aug[i] = input(i,j);
    }
    for(int i = 0; i < m_num_recurrent; i++)
    {
      input_aug[i+m_num_in] = prev_output[i];
    }

    output_aug = (m_weights * input_aug) + m_biases;

    for(int i = 0; i < m_num_out; i++)
    {
      output(i,j) = CppAD::tanh(output_aug[i]);
    }
    for(int i = 0; i < m_num_recurrent; i++)
    {
      prev_output[i] = CppAD::tanh(output_aug[i+m_num_out]);
    }
  }


}

template<typename Scalar>
void RecurrentLayer<Scalar>::setParams(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
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
unsigned RecurrentLayer<Scalar>::getNumParams()
{
  return (m_weights.rows()*m_weights.cols()) + m_biases.rows();
}


template<typename Scalar>
void RecurrentLayer<Scalar>::getParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
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



template class RecurrentLayer<float>;
template class RecurrentLayer<ADF>;
template class RecurrentLayer<ADAD>;
