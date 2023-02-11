#include "NormLayer.h"
#include <limits.h>
#include "gtest/gtest.h"
#include "types/Scalars.h"

class NormLayerFixture
{
public:
  static Eigen::Matrix<ADF, Eigen::Dynamic, 1> getMean(NormLayer<ADF> &norm_layer)
  {
    return norm_layer.m_mean;
  }
  static Eigen::Matrix<ADF, Eigen::Dynamic, 1> getStddev(NormLayer<ADF> &norm_layer)
  {
    return norm_layer.m_stddev;
  }

};


namespace {

  TEST(NormLayerFixture, NormLayer){
    int len = 1000;
    int dim = 2;
    Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> input(2,len);
    Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> output(2,len);
    NormLayer<ADF> norm_layer(dim,dim);

    for(int i = 0; i < len; i++)
      {
        input(0,i) = 10;
        input(1,i) = (10.0f * i)/len;
      }

    norm_layer.process(output, input);

    Eigen::Matrix<ADF, Eigen::Dynamic, 1> mean;
    Eigen::Matrix<ADF, Eigen::Dynamic, 1> stddev;
    mean = NormLayerFixture::getMean(norm_layer);
    stddev = NormLayerFixture::getStddev(norm_layer);

    for(int i = 0; i < dim; i++)
      {
        printf("%f ", CppAD::Value(mean[i]));
      }
    printf("\n");

    for(int i = 0; i < dim; i++)
      {
        printf("%f ", CppAD::Value(stddev[i]));
      }
    printf("\n");

    EXPECT_NEAR(10.0f, CppAD::Value(mean[0]), 1e-3);
    EXPECT_NEAR(9.0f,  CppAD::Value(mean[1]), .5);
    EXPECT_NEAR(0.0f,  CppAD::Value(stddev[0]), 1e-3);

  }


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}
