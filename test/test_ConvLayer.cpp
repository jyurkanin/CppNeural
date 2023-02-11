#include "ConvLayer.h"
#include <limits.h>
#include "gtest/gtest.h"

class ConvLayerFixture
{
public:
  static void initWeights(ConvLayer<ADF>& layer)
  {
    for(int k = 0; k < layer.m_filters.size(); k++)
    {
      for(int i = 0; i < layer.m_filters[k].rows(); i++)
      {
        for(int j = 0; j < layer.m_filters[k].cols(); j++)
        {
          layer.m_filters[k](i,j) = 1;
        }
      }
    }
  }

  static void zeroBias(ConvLayer<ADF>& layer)
  {
    for(int i = 0; i < layer.m_biases.size(); i++)
    {
      layer.m_biases[i] = 0;
    }
  }
};

namespace {

  TEST(ConvLayerFixture, ConvLayer)
  {
    int len = 1000;
    int filter_size = 16;
    Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> input(1,len);
    Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> output(2,len);
    ConvLayer<ADF> conv_layer(1,2,filter_size,1);
    ConvLayerFixture::initWeights(conv_layer);

    for(int i = 0; i < len; i++)
    {
      input(0,i) = 1;
    }

    conv_layer.process(output, input);

    for(int i = 0; i < filter_size-1; i++)
    {
      EXPECT_EQ(0.0f,  CppAD::Value(output(0,i)));
      EXPECT_EQ(0.0f,  CppAD::Value(output(1,i)));
    }

    EXPECT_NEAR(16, CppAD::Value(output(0,filter_size+0)), 1e-3f);
    EXPECT_NEAR(16, CppAD::Value(output(0,filter_size+1)), 1e-3f);
    EXPECT_NEAR(16, CppAD::Value(output(0,filter_size+2)), 1e-3f);
    EXPECT_NEAR(16, CppAD::Value(output(0,len-1)),         1e-3f);

    EXPECT_NEAR(16, CppAD::Value(output(1,filter_size+0)), 1e-3f);
    EXPECT_NEAR(16, CppAD::Value(output(1,filter_size+1)), 1e-3f);
    EXPECT_NEAR(16, CppAD::Value(output(1,filter_size+2)), 1e-3f);
    EXPECT_NEAR(16, CppAD::Value(output(1,len-1)),         1e-3f);

  }

  TEST(ConvLayerFixture, Dilation)
  {
    int len = 100;
    Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> input(1,len);
    Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> output(1,len);

    ConvLayer<ADF> conv_layer(1,1,2,2);
    ConvLayerFixture::initWeights(conv_layer);
    ConvLayerFixture::zeroBias(conv_layer);

    for(int i = 0; i < len; i++)
    {
      input(0,i) = 1;
    }

    conv_layer.process(output, input);

    // for(int i = 0; i < len; i++)
    // {
    //   printf("%.2f ", CppAD::Value(output(0,i)));
    // }
    // printf("\n");

    EXPECT_NEAR(CppAD::Value(output(0,0)), 0, 1e-3f);
    EXPECT_NEAR(CppAD::Value(output(0,1)), 0, 1e-3f);
    EXPECT_NEAR(CppAD::Value(output(0,2)), 2, 1e-3f);
  }


}
