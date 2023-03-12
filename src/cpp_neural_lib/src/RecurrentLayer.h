#include <Eigen/Core>
#include "types/Scalars.h"
#include "BaseLayer.h"

#pragma once

//class RecurrentLayerFixture;
template<typename Scalar>
class RecurrentLayer : public BaseLayer<Scalar>
{
 public:
  RecurrentLayer(int num_in, int num_recurrent, int num_out);
  ~RecurrentLayer();

  void process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &controls, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input);
  void getParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx);
  void setParams(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx);
  unsigned getNumParams();

 private:
  //friend class RecurrentLayerFixture;

  int m_num_in;
  int m_num_out;
  int m_num_recurrent;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m_weights;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_biases;
};
