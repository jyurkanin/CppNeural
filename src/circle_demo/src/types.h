#include <cppad/cppad.hpp>
#include <Eigen/Core>

//typedef CppAD::cg::CG<float> CGF;                                                                                                       
//typedef CppAD::AD<CGF> ADCF;                                                                                                            
//typedef CppAD::AD<ADCF> ADAD;                                                                                                           
typedef CppAD::AD<float> ADF;

typedef Eigen::Matrix<ADF,   Eigen::Dynamic, 1> VectorAD;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorF;
typedef Eigen::Matrix<ADF,   Eigen::Dynamic, Eigen::Dynamic> MatrixAD;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixF;
