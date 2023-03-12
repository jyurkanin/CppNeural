#include "FinalLayer.h"


template<typename Scalar>
FinalLayer<Scalar>::FinalLayer() : BaseLayer<Scalar>(1,2)
{
    m1 = 4;
}

template<typename Scalar>
FinalLayer<Scalar>::~FinalLayer()
{
  
}

template<typename Scalar>
unsigned FinalLayer<Scalar>::getClassNumParams()
{
  return 1;
}

template<typename Scalar>
unsigned FinalLayer<Scalar>::getNumParams()
{
  return getClassNumParams();
}

template<typename Scalar>
void FinalLayer<Scalar>::process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &controls, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input)
{
    //Just going to softmax here.
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> soft_max_classes(input.rows());
    for(int i = 0; i < controls.cols(); i++)
    {
      Scalar sum(0.0f);
      for(int j = 0; j < input.rows(); j++)
      {
        Scalar temp = .5*m1*(1+input(j,i));
        soft_max_classes[j] = CppAD::exp(temp);
        sum += soft_max_classes[j];
      }

      controls(0,i) = soft_max_classes[0] / sum;
      controls(1,i) = soft_max_classes[1] / sum;
      //Don't do anything with soft_max_classes[2] because
      //that just means hold which requires no actions.
    }
}


template<typename Scalar>
void FinalLayer<Scalar>::getParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
{
    params[idx+0] = m1;
    idx += getNumParams();
}

template<typename Scalar>
void FinalLayer<Scalar>::setParams(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
{
    m1 = params[idx+0];
    idx += getNumParams();
}


template class FinalLayer<float>;
template class FinalLayer<ADF>;
template class FinalLayer<ADAD>;
