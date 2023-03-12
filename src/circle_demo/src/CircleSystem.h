#include <cpp_bptt.h>
#include <cpp_neural.h>

#include "Network.h"

template<typename Scalar>
class CircleSystem : public cpp_bptt::System<Scalar>
{
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
  CircleSystem();
  ~CircleSystem();
  
  virtual void   setParams(const VectorS &params);
  virtual void   getParams(VectorS &params);
  virtual void   forward(const VectorS &X, VectorS &Xd);
  virtual Scalar loss(const VectorS &gt_vec, VectorS &vec);

private:
  Scalar          m_target_radius{0.1};
  Network<Scalar> m_network;
};

