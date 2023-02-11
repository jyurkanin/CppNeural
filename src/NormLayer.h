#include <Eigen/Core>
#include "types/Scalars.h"
#include "BaseLayer.h"

#pragma once

//fwd dcl of test fixture
class NormLayerFixture;

template<typename Scalar>
class NormLayer : public BaseLayer<Scalar>
{
 public:
  NormLayer(int num_in, int num_out);
  virtual ~NormLayer();

  virtual void process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &controls, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input);
  virtual void getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx);
  virtual void setParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx);
  virtual unsigned getNumParams();
  virtual void reset();
  virtual void zeroBias();
  
 private:
  friend class NormLayerFixture; //for gtesting purposes

  int m_is_init;
  Scalar m_beta;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_mean;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_stddev;
};

template<>
void NormLayer<ADF>::getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx);
