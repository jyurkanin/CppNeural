#include <cpp_neural.h>
#include <types/Tensors.h>
#include <iostream>


template<typename Scalar>
class Network : public BaseLayer<Scalar>
{
public:
  Network()
      : BaseLayer<Scalar>(2, 2),
	m_num_in(2),
	m_num_hidden(4),
	m_num_out(2),
        m_dense_layer1(2 + 3, 4),
	m_dense_layer2(4, 2) {}
  Network(int num_in, int num_hidden, int num_out)
      : BaseLayer<Scalar>(num_in, num_out),
	m_num_in(num_in),
        m_num_hidden(num_hidden),
	m_num_out(num_out),
        m_dense_layer1(num_in + 3, num_hidden),
        m_dense_layer2(num_hidden, num_out) {}
  ~Network() {}

  void process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &controls, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input)
  {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> extended_input(input.rows()+3, input.cols());// = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.rows()+3, input.cols());
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> layer1_out(m_dense_layer1.getNumOutputs(), input.cols());
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> layer2_out(m_dense_layer2.getNumOutputs(), input.cols());

    for(int i = 0; i < input.rows(); i++)
    {
      for(int j = 0; j < input.cols(); j++)
      {
	extended_input(i,j) = input(i,j);
	extended_input(i+2,j) = input(i,j)*input(i,j);
      }
    }

    for(int j = 0; j < input.cols(); j++)
    {
      extended_input(4,j) = input(0,j)*input(1,j);
    }
    
    m_dense_layer1.process(layer1_out, extended_input);      helpers::applyTanh(layer1_out);
    m_dense_layer2.process(controls, layer1_out);            //helpers::applyTanh(controls);
  }
  
  void getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx)
  {
    m_dense_layer1.getParams(params, idx);
    m_dense_layer2.getParams(params, idx);
  }
  
  void setParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx)
  {
    m_dense_layer1.setParams(params, idx);
    m_dense_layer2.setParams(params, idx);
  }
  unsigned getNumParams()
  {
    return m_dense_layer1.getNumParams() + m_dense_layer2.getNumParams();
  }
  void zeroBias() {}

private:
  int m_num_in;
  int m_num_out;
  int m_num_hidden;
  
  DenseLayer<Scalar> m_dense_layer1;
  DenseLayer<Scalar> m_dense_layer2;
};


