#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "Generator.h"
#include "Simulator.h"
#include "SimulatorCodeGen.h"

using namespace CppAD;
using namespace CppAD::cg;



int generate_sources(){
  Eigen::Matrix<ADAD,Eigen::Dynamic,1> y_adad(1);
  Eigen::Matrix<ADAD,Eigen::Dynamic,1> x_init(2);
  Eigen::Matrix<ADAD,Eigen::Dynamic,1> theta(6);
  Eigen::Matrix<ADAD,Eigen::Dynamic,1> x_adad(theta.size() + x_init.size());
  
  //Generate second order template autodiff function
  Independent(x_adad);
  for(int i = 0; i < theta.size(); i++){
    theta[i] = x_adad[i];
  }
  
  for(int i = theta.size(); i < (theta.size() + x_init.size()); i++){
    x_init[i-theta.size()] = x_adad[i];
  }
  
  solve_loss(x_init, theta, y_adad, 100);
  ADFun<ADCF> fun2(x_adad, y_adad);
  
  printf("Created 2nd order function\n");
  
  //Generate first order autodiff function.
  //This function won't be autodiffed. It will only be used in
  //zero order forward mode. This function is useful though
  //because it selects which components of the derivative to keep.
  //Also it performs zero order forward mode of the original loss function.
  //So hopefully by combining the derivative and the forward mode we can
  //get better effiency than if both were calculated separately.
  
  Eigen::Matrix<ADCF,Eigen::Dynamic,1> x_ad(x_adad.size());
  Eigen::Matrix<ADCF,Eigen::Dynamic,1> y_ad(theta.size()+1); //size of gradient plus loss
  Independent(x_ad);

  Eigen::Matrix<ADCF,Eigen::Dynamic,1> y1(1); //partial wrt self
  y1[0] = 1;
  
  Eigen::Matrix<ADCF,Eigen::Dynamic,1> loss = fun2.Forward(0,x_ad);
  Eigen::Matrix<ADCF,Eigen::Dynamic,1> grad = fun2.Reverse(1,y1);
  
  printf("errno %d  EDOM=%d\n", errno, EDOM);
  
  for(int i = 0; i < theta.size(); i++){
    y_ad[i] = grad[i];
  }
  
  y_ad[theta.size()] = loss[0];
  
  ADFun<CGF> fun1(x_ad, y_ad);
  printf("Created 1st order function\n");
  
  fun1.optimize();
  printf("Optimized the function successfully.\n");
  
  CodeHandler<float> handler;
  handler.setVerbose(true);

  int ind_size = theta.size() + x_init.size();
  CppAD::vector<CGF> indVars(ind_size); //corresponds to theta
  handler.makeVariables(indVars);
  
  CppAD::vector<CGF> y0 = fun1.Forward(0, indVars);
  
  LanguageC<float> langC("float");
  LangCDefaultVariableNameGenerator<float> nameGen;
  
  std::ostringstream code;
  handler.generateCode(code, langC, y0, nameGen);
  //std::cout << code.str();

  std::ofstream output_file("system.cpp");
  output_file << code.str();
  output_file.close();
}
