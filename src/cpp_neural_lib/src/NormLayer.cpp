#include "NormLayer.h"
#include <cassert>


//constexpr NormLaer::m_beta = .99f;
template<typename Scalar>
NormLayer<Scalar>::NormLayer(int num_in, int num_out) : BaseLayer<Scalar>(num_in, num_out)
{
  assert(num_in == num_out);

  m_mean = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(num_in);
  m_stddev = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(num_in);
  m_is_init = 0;

  m_beta = Scalar(.99f);
}

template<typename Scalar>
NormLayer<Scalar>::~NormLayer(){}


template<typename Scalar>
void NormLayer<Scalar>::process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &output,
                        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input)
{
  assert(input.rows() == this->getNumInputs());
  assert(output.rows() == this->getNumOutputs());

  if(!m_is_init)
  {
    m_is_init = 1;
    m_mean = input.col(0);
    m_stddev = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(input.rows());
  }

  for(int i = 0; i < input.cols(); i++)
  {
    m_mean = (m_beta*m_mean) + (1-m_beta)*input.col(i);
    m_stddev = (m_beta*m_stddev.array().square()) + (1-m_beta)*(m_mean - input.col(i)).array().square();
    m_stddev = m_stddev.array().sqrt();

    output.col(i) = (input.col(i) - m_mean).array() / m_stddev.array();
  }

  //printf("%f\n", CppAD::Value(m_mean[i]));
}


template<typename Scalar>
void NormLayer<Scalar>::setParams(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx){}

template<typename Scalar>
unsigned NormLayer<Scalar>::getNumParams(){return 0;}

template<typename Scalar>
void NormLayer<Scalar>::reset(){m_is_init = 0;}

template<typename Scalar>
void NormLayer<Scalar>::zeroBias(){}



template<typename Scalar>
void NormLayer<Scalar>::getParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
{

}

template class NormLayer<double>;
template class NormLayer<ADF>;
template class NormLayer<ADAD>;
