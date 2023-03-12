#include "ConvLayer.h"
#include <assert.h>
#include <cstdlib>

template<typename Scalar>
ConvLayer<Scalar>::ConvLayer(int num_in_channels, int num_filters, int filter_size, int dilation) : BaseLayer<Scalar>(num_in_channels, num_filters)
{
  m_dilation = dilation;
  m_num_filters = num_filters;
  m_filter_size = filter_size;
  m_num_in_channels = num_in_channels;
  
  Scalar small(.01);
  
  for(int i = 0; i < num_filters; i++)
  {
    m_filters.push_back(small*Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(num_in_channels, filter_size));
  }
  
  m_biases = small*Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Random(num_filters);

  for(int k = 0; k < m_num_filters; k++)
  {
    for(int i = 0; i < m_num_in_channels; i++)
    {
      for(int j = 0; j < m_filter_size; j++)
      {
        m_filters[k](i,j) = m_filters[k](i,j) - (.5f*small);
      }
    }
  }
}

template<typename Scalar>
ConvLayer<Scalar>::~ConvLayer()
{

}

template<typename Scalar>
void ConvLayer<Scalar>::process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &output, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input)
{
  
  int num_samples = input.cols();
  int num_in_channels = input.rows();
  int dilated_filter_size = ((m_filter_size-1) * m_dilation) + 1;

  assert(input.rows() == m_num_in_channels);
  assert(output.cols() == input.cols());
  assert(output.rows() == m_num_filters);
  assert(num_samples >= m_filter_size);

  for(int j = 0; j < m_num_filters; j++)
  {
    for(int k = 0; k < dilated_filter_size-1; k++)
    {
      output(j, k) = 0; //instead of zero padding, just set these intial values to zero.
    }
    
    //run filter over entire input
    for(int k = dilated_filter_size-1; k < num_samples; k++)
    {
      Scalar sum(0.0f);
      int start_idx = k - (dilated_filter_size - 1);

      for(int i = 0; i < m_num_in_channels; i++)
      {
        for(int m = 0; m < m_filter_size; m++)
        {
          sum += input(i, start_idx + (m * m_dilation)) * m_filters[j](i, m);
        }
      }

      output(j, k) = sum + m_biases[j]; //dot product and bias term
    }
  }
}

template<typename Scalar>
void ConvLayer<Scalar>::setParams(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
{
  for(int k = 0; k < m_num_filters; k++)
  {
    for(int i = 0; i < m_num_in_channels; i++)
    {
      for(int j = 0; j < m_filter_size; j++)
      {
        m_filters[k](i,j) = params[idx];
        idx++;
      }
    }
  }

  for(int i = 0; i < m_num_filters; i++)
  {
    m_biases[i] = params[idx];
    idx++;
  }
}

template<typename Scalar>
unsigned ConvLayer<Scalar>::getNumParams()
{
  return (m_num_filters * m_num_in_channels * m_filter_size) + m_num_filters;
}

template<typename Scalar>
unsigned ConvLayer<Scalar>::getFilterSize()
{
  return m_filter_size;
}

template<typename Scalar>
unsigned ConvLayer<Scalar>::getNumFilters()
{
  return m_num_filters;
}

template<typename Scalar>
unsigned ConvLayer<Scalar>::getNumChannels()
{
  return m_num_in_channels;
}


template<typename Scalar>
const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& ConvLayer<Scalar>::getFilters()
{
  return m_filters;
}


template<typename Scalar>
void ConvLayer<Scalar>::getParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
{
  for(int k = 0; k < m_num_filters; k++)
  {
    for(int i = 0; i < m_num_in_channels; i++)
    {
      for(int j = 0; j < m_filter_size; j++)
      {
        params[idx] = m_filters[k](i,j);
        idx++;
      }
    }
  }

  for(int i = 0; i < m_num_filters; i++)
  {
    params[idx] = m_biases[i];
    idx++;
  }
}

template<typename Scalar>
void ConvLayer<Scalar>::zeroBias()
{
  Scalar zero(0);
  for(int i = 0; i < m_biases.size(); i++)
  {
    m_biases[i] = zero;
  }
}


template class ConvLayer<ADF>;
template class ConvLayer<ADAD>;
