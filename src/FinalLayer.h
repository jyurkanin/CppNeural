#include <Eigen/Core>
#include "types/Scalars.h"
#include "BaseLayer.h"


//I've decided this is Just going to compute a softmax.
template<typename Scalar>
class FinalLayer : BaseLayer<Scalar>
{
 public:
  FinalLayer();
  ~FinalLayer();
  
  void process(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &controls, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input);
  void getParams(Eigen::Matrix<float, Eigen::Dynamic, 1>& params, int &idx);
  void setParams(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params, int &idx);
  unsigned getNumParams();
  void zeroBias(){}
  
  static unsigned getClassNumParams();

  Scalar m1;
};
