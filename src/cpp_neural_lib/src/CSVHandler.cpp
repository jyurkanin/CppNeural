#include "CSVHandler.h"


namespace CSV{
  void load_csv(std::string fn, Eigen::Matrix<double,Eigen::Dynamic,1> &params){
    std::ifstream csv_file(fn);
    
    char comma;
    for(int i = 0; i < params.size(); i++){
      csv_file >> params[i];
      csv_file >> comma;
    }
    
    csv_file.close();
  }
  
  void write_csv(std::string fn, const Eigen::Matrix<double,Eigen::Dynamic,1> &params){
    std::ofstream csv_file(fn);
    
    char comma = ',';
    for(int i = 0; i < params.size(); i++){
      csv_file << params[i];
      csv_file << comma;
    }
    
    csv_file.close();
  }
  
}


