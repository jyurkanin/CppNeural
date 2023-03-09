#include <cpp_bptt.h>
#include <cpp_neural.h>

template <typename Scalar>
class CircleSystem<Scalar> : public cpp_bptt::System<Scalar>
{
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
  CircleSystem();
  ~CircleSystem();
  
  virtual void   setParams(const VectorS &params);
  virtual int    getNumParams();
  virtual void   forward(const VectorS &X, VectorS &Xd);
  virtual Scalar loss(const VectorS &gt_vec, VectorS &vec);

private:
  Network<Scalar> m_network;
};

