#include <Eigen/Core>
#include "types/Scalars.h"

namespace helpers
{

template<typename Scalar>
void getSpread(Scalar price, Scalar &spread_buy, Scalar &spread_sell);

template<typename Scalar>
void applyTanh(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat);

template<typename Scalar>
void applyRelu(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat);

template<typename Scalar>
Scalar computeNorm(Eigen::Matrix<Scalar, Eigen::Dynamic,1> &vec);

}
