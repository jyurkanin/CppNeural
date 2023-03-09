#include <Eigen/Dense>
#include <string>
#include <fstream>

namespace CSV{
  void load_csv(std::string fn, Eigen::Matrix<float,Eigen::Dynamic,1> &params);
  void write_csv(std::string fn, const Eigen::Matrix<float,Eigen::Dynamic,1> &params);
  
}
