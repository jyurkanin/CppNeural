#include "System.h"
#include <types/Scalars.h>
#include <types/Tensors.h>

namespace cpp_bptt
{

template <typename Scalar>
class SimpleSystem : public System<Scalar>
{
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
  SimpleSystem();
  ~SimpleSystem();
  
  virtual void   setParams(const VectorS &params);
  virtual void   getParams(VectorS &params);
  virtual void   forward(const VectorS &X, VectorS &Xd);
  virtual Scalar loss(const VectorS &gt_vec, VectorS &vec);
  
  
private:
  MatrixS m_params;
};

} //cpp_bptt
