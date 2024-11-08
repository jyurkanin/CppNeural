#include "helpers.h"


namespace helpers
{



template<typename Scalar>
void applyTanh(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
  for(unsigned i = 0; i < mat.rows(); i++)
    {
      for(unsigned j = 0; j < mat.cols(); j++)
        {
          mat(i,j) = CppAD::tanh(mat(i,j));
        }
    }

}

template void applyTanh(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat);
template void applyTanh(Eigen::Matrix<ADF,   Eigen::Dynamic, Eigen::Dynamic> &mat);
template void applyTanh(Eigen::Matrix<ADAD,  Eigen::Dynamic, Eigen::Dynamic> &mat);



template<typename Scalar>
void applyRelu(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
  Scalar zero;
  for(unsigned i = 0; i < mat.rows(); i++)
    {
      for(unsigned j = 0; j < mat.cols(); j++)
        {
          mat(i,j) = CppAD::CondExpGe(mat(i,j), zero, mat(i,j), zero);
        }
    }
}

template void applyRelu(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat);
template void applyRelu(Eigen::Matrix<ADF,   Eigen::Dynamic, Eigen::Dynamic> &mat);
template void applyRelu(Eigen::Matrix<ADAD,  Eigen::Dynamic, Eigen::Dynamic> &mat);


template<typename Scalar>
Scalar computeNorm(Eigen::Matrix<Scalar, Eigen::Dynamic,1> &vec)
{
  Scalar sum{0};

  for(int i = 0; i < vec.rows(); i++)
  {
    sum += vec[i]*vec[i];
  }

  return CppAD::sqrt(sum);
}

template double computeNorm(Eigen::Matrix<double, Eigen::Dynamic,1> &vec);
template ADF   computeNorm(Eigen::Matrix<ADF,   Eigen::Dynamic,1> &vec);
template ADAD  computeNorm(Eigen::Matrix<ADAD,  Eigen::Dynamic,1> &vec);

  
}
