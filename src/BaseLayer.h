#pragma once

#include <Eigen/Core>
#include "types/Scalars.h"

template<typename Scalar>
class BaseLayer
{
 public:
  BaseLayer(int num_in, int num_out);
  virtual ~BaseLayer();

  virtual void process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &controls, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input) = 0;
  virtual void getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx) = 0;
  virtual void setParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx) = 0;
  virtual unsigned getNumParams() = 0;
  virtual void zeroBias() = 0;
  
  virtual void reset();

  int getNumInputs();
  int getNumOutputs();

 private:
  int m_num_in;
  int m_num_out;

};
