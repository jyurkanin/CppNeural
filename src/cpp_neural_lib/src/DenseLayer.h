#include <Eigen/Core>
#include "types/Scalars.h"
#include "BaseLayer.h"

#pragma once

class AgentTestFixture;

//class DenseLayerFixture;
template<typename Scalar>
class DenseLayer : public BaseLayer<Scalar>
{
 public:
  DenseLayer(int num_in, int num_out);
  ~DenseLayer();

  void process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &controls, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input);
  void getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx);
  void setParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx);
  unsigned getNumParams();
  void zeroBias();
  
 private:
  //friend class DenseLayerFixture;
  friend class AgentTestFixture;

  int m_num_in;
  int m_num_out;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m_weights;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_biases;
};

template<>
void DenseLayer<ADF>::getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx);
