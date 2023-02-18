#include <Eigen/Core>
#include "types/Scalars.h"
#include "BaseLayer.h"

#include <utility>

#pragma once

class AgentTestFixture;
class ConvLayerFixture;

template<typename Scalar>
class ConvLayer : public BaseLayer<Scalar>
{
 public:
  ConvLayer(int num_in_channels, int num_filters, int filter_size, int dilation);
  ~ConvLayer();

  void process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &controls, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input);
  void getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx);
  void setParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx);
  void zeroBias();
  
  unsigned getNumParams();
  unsigned getFilterSize();
  unsigned getNumFilters();
  unsigned getNumChannels();
  
  const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& getFilters();
  
  std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> m_filters;
  
 private:
  friend class ConvLayerFixture;
  friend class AgentTestFixture;

  int m_dilation;
  int m_num_filters;
  int m_filter_size;
  int m_num_in_channels;
  
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_biases;
};

template<>
void ConvLayer<ADF>::getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx);
