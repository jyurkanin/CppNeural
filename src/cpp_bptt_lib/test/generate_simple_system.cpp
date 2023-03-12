#include <memory>
#include <fenv.h>
#include <iostream>

#include "Generator.h"
#include "SimpleSystem.h"
#include "SimulatorADAD.h"
#include <types/Scalars.h>
#include <types/Tensors.h>


int main2()
{
  std::shared_ptr<cpp_bptt::SimpleSystem<ADF>> system = std::make_shared<cpp_bptt::SimpleSystem<ADF>>();

  VectorAD xk1_adf(system->getStateDim());
  VectorAD theta_adf(system->getNumParams());
  VectorAD xk1_gt_adf(system->getStateDim());
  VectorAD loss_params_dynamic_vars_adf(xk1_adf.size() + xk1_gt_adf.size());
  VectorAD loss_adf(1);
  
  CppAD::Independent(theta_adf, loss_params_dynamic_vars_adf);
  
  int idx = 0;
  for(int i = 0; i < xk1_adf.size(); i++)
  {
    xk1_adf[i] = loss_params_dynamic_vars_adf[idx+i];
  }
  idx += xk1_adf.size();
  for(int i = 0; i < xk1_gt_adf.size(); i++)
  {
    xk1_gt_adf[i] = loss_params_dynamic_vars_adf[idx+i];
  }
  idx += xk1_gt_adf.size();
  
  system->setParams(theta_adf);
  loss_adf[0] = system->loss(xk1_gt_adf, xk1_adf);
  CppAD::ADFun<float> partial_loss_params_adf(theta_adf, loss_adf);
  partial_loss_params_adf.optimize(); //Probably will segfault.

  VectorF y0_f(1); y0_f[0] = 1;
  VectorF theta_f = 2*VectorF::Ones(system->getNumParams());
  VectorF loss_params_dynamic_vars_f = VectorF::Ones(xk1_adf.size() + xk1_gt_adf.size());
  
  partial_loss_params_adf.new_dynamic(loss_params_dynamic_vars_f);
  VectorF loss = partial_loss_params_adf.Forward(0, theta_f);
  VectorF loss_params_gradient = partial_loss_params_adf.Reverse(1, y0_f);

  std::cout << "Loss " << loss[0] << "\n";
  for(int i = 0; i < loss_params_gradient.size(); i++)
  {
    std::cout << loss_params_gradient[i] << "\n";
  }

  return 0;
}

int main(){
  // feenableexcept(FE_INVALID | FE_OVERFLOW);
  std::shared_ptr<cpp_bptt::SimpleSystem<ADAD>> system = std::make_shared<cpp_bptt::SimpleSystem<ADAD>>();
  std::shared_ptr<cpp_bptt::SimulatorADAD> simulator = std::make_shared<cpp_bptt::SimulatorADAD>(system);
  
  cpp_bptt::Generator generator;
  generator.setSimulator(simulator);
  generator.initialize();

  return 0;
}
