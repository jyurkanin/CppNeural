#pragma once

#include "Scalars.h"
#include <Eigen/Core>

typedef Eigen::Matrix<ADAD,  Eigen::Dynamic, 1>  VectorADAD;
typedef Eigen::Matrix<ADCF,  Eigen::Dynamic, 1>  VectorADCF;
typedef Eigen::Matrix<CGF,   Eigen::Dynamic, 1>  VectorCGF;
typedef Eigen::Matrix<ADF,   Eigen::Dynamic, 1>  VectorAD;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorF;

typedef Eigen::Matrix<ADAD,  Eigen::Dynamic, Eigen::Dynamic>  MatrixADAD;
typedef Eigen::Matrix<ADCF,  Eigen::Dynamic, Eigen::Dynamic>  MatrixADCF;
typedef Eigen::Matrix<CGF,   Eigen::Dynamic, Eigen::Dynamic>  MatrixCGF;
typedef Eigen::Matrix<ADF,   Eigen::Dynamic, Eigen::Dynamic>  MatrixAD;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixF;
