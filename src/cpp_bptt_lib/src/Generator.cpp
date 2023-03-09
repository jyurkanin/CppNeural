#include "Generator.h"

#include <vector>



void Generator::initialize()
{
  generatePartialLossState();
  generatePartials();
  // generatePartialStateParams();
  // generatePartialStatePrevState();
}

void Generator::generatePartialLossState()
{
  std::shared_ptr<System<ADAD>> system = m_simulator->getSystem();
  
  // 1st. Use ADAD to compute an ADCF function
  VectorADAD x0(system->getStateDim());
  VectorADAD x0_gt(system->getStateDim());
  VectorADAD loss(1);
      
  CppAD::Independent(x0, x0_gt);
  loss[0] = system->loss(x0_gt, x0);
  
  CppAD::ADFun<ADCF> partial_loss_state_adcf(x0, loss);
  partial_loss_state_adcf.optimize(); //Probably will segfault.
  
  
  // 2nd. Use the ADCF function's reverse and forward to compute a CGF forward function
  VectorADCF x0_gt_adcf(system->getStateDim());
  VectorADCF x0_adcf(system->getStateDim());
  VectorADCF x_all_adcf(system->getStateDim()*2);
  
  CppAD::Independent(x_all_adcf);
  for(int i = 0; i < x0_adcf.size(); i++)
  {
    x0_adcf[i] = x_all_adcf[i];
    x0_gt_adcf[i] = x_all_adcf[i + x0_adcf.size()];
  }
  
  VectorADCF y1_adcf(1);
  VectorADCF loss_adcf(1);
  VectorADCF grad_adcf(system->getStateDim()); // partial loss state
  VectorADCF y_all_adcf(system->getStateDim() + 1);
  
  y1_adcf[0] = 1; //boilerplate basically
  
  partial_loss_state_adcf.new_dynamic(x0_gt_adcf);
  loss_adcf = partial_loss_state_adcf.Forward(0, x0_adcf);
  grad_adcf = partial_loss_state_adcf.Reverse(1, y1_adcf);
  for(int i = 0; i < grad_adcf.size(); i++)
  {
    y_all_adcf[i] = grad_adcf[i];
  }
  y_all_adcf[grad_adcf.size()] = loss_adcf[0];
  
  CppAD::ADFun<CGF> func_final(x_all_adcf, y_all_adcf);
  func_final.optimize();
  
  // 3rd. Now to actually generate the fucking code :(  
  std::vector<CGF> input_vars(system->getStateDim()*2); // x0_gt.size() + x0.size()
  std::vector<CGF> output_vars(y_all_adcf.size());  // gradient size+ loss
  
  CppAD::cg::CodeHandler<float> handler;
  handler.setVerbose(true);
  handler.makeVariables(input_vars);
  output_vars = func_final.Forward(0, input_vars);
  
  CppAD::cg::LanguageC<float> langC("float");
  CppAD::cg::LangCDefaultVariableNameGenerator<float> name_gen;
  std::ostringstream code;
  handler.generateCode(code, langC, output_vars, name_gen);
  
  std::ofstream output_file("partial_loss_state.cpp");
  output_file <<
    "computePartialLossState(const VectorF &gt_x1,\n"
    "                        const VectorF &x1,\n"
    "                        VectorF &loss_x1_partial,\n"
    "                        float &loss)\n"
    "{\n";
  output_file << "  float x[" << input_vars.size() <<"];\n";
  output_file << "  float y[" << output_vars.size() <<"];\n";
  output_file <<
    "  int idx = 0;\n"
    "  for(int i = 0; i < gt_x1.size(); i++)\n"
    "  {\n"
    "    x[i+idx] = x1[i];\n"
    "  }\n"
    "  idx += gt_x1.size();\n";
  output_file <<
    "  for(int i = 0; i < x1.size(); i++)\n"
    "  {\n"
    "    x[i+idx] = gt_x1[i];\n"
    "  }\n"
    "  idx += x1.size();\n\n";

  output_file << code.str();
  output_file <<
    "\n"
    "  for(int i = 0; i < loss_x1_partial.size(); i++)\n"
    "  {\n"
    "    loss_x1_partial[i] = y[i];\n"
    "  }\n"
    "  loss = y[loss_x1_partial.size()]\n"
    "}";
  output_file.close();
}

void Generator::generatePartials()
{
  std::shared_ptr<System<ADAD>> system = m_simulator->getSystem();
  
  // 1.1. Use ADAD to compute an ADCF function
  VectorADAD xk0_adad(system->getStateDim());
  VectorADAD xk1_adad(system->getStateDim());
  VectorADAD theta_adad(system->getNumParams());
  
  CppAD::Independent(xk0_adad, theta_adad);
  system->setParams(theta_adad);
  m_simulator->integrate(xk0_adad, xk1_adad);
  
  CppAD::ADFun<ADCF> partial_state_prev_state_adcf(xk0_adad, xk1_adad);
  partial_state_prev_state_adcf.optimize(); //Probably will segfault.
  
  // 1.2. Use ADAD to compute another ADCF function
  // VectorADAD xk0_adad(system->getStateDim());
  // VectorADAD xk1_adad(system->getStateDim());
  // VectorADAD theta_adad(system->getNumParams());
  
  CppAD::Independent(theta_adad, xk0_adad);
  system->setParams(theta_adad);
  m_simulator->integrate(xk0_adad, xk1_adad);
  
  CppAD::ADFun<ADCF> partial_state_params_adcf(theta_adad, xk1_adad);
  partial_state_params_adcf.optimize(); //Probably will segfault.
  
  
  // 2nd. Use the ADCF function's reverse and forward to compute a CGF forward function
  VectorADCF xk0_adcf(system->getStateDim());
  VectorADCF theta_adcf(system->getNumParams());
  VectorADCF x_all_adcf(xk0_adcf.size() + theta_adcf.size());
  
  CppAD::Independent(x_all_adcf);
  for(int i = 0; i < xk0_adcf.size(); i++)
  {
    xk0_adcf[i] = x_all_adcf[i];
  }
  for(int i = 0; i < theta_adcf.size(); i++)
  {
    theta_adcf[i] = x_all_adcf[i + xk0_adcf.size()];
  }
  
  VectorADCF xk1_adcf(system->getStateDim());
  VectorADCF state_state_jacobian_adcf(system->getStateDim() * system->getStateDim());
  VectorADCF state_param_jacobian_adcf(system->getStateDim() * system->getNumParams());
  VectorADCF y_all_adcf(xk1_adcf.size() +
			state_state_jacobian_adcf.size() +
			state_param_jacobian_adcf.size());
  
  partial_state_prev_state_adcf.new_dynamic(theta_adcf);
  xk1_adcf = partial_state_prev_state_adcf.Forward(0, xk0_adcf);
  state_state_jacobian_adcf = partial_state_prev_state_adcf.Jacobian(xk0_adcf);
  
  partial_state_params_adcf.new_dynamic(xk0_adcf);
  state_param_jacobian_adcf = partial_state_params_adcf.Jacobian(theta_adcf);

  int idx = 0;
  for(int i = 0; i < xk1_adcf.size(); i++)
  {
    y_all_adcf[i] = xk1_adcf[i];  
  }
  idx += xk1_adcf.size();
  
  for(int i = 0; i < state_state_jacobian_adcf.size(); i++)
  {
    y_all_adcf[idx + i] = state_state_jacobian_adcf[i];
  }
  idx += state_state_jacobian_adcf.size();

  for(int i = 0; i < state_param_jacobian_adcf.size(); i++)
  {
    y_all_adcf[idx + i] = state_param_jacobian_adcf[i];
  }
  idx += state_param_jacobian_adcf.size();
  
  CppAD::ADFun<CGF> func_final(x_all_adcf, y_all_adcf);
  func_final.optimize();
  
  
  // 3rd. Now to actually generate the fucking code :(  
  std::vector<CGF> input_vars(x_all_adcf.size());
  std::vector<CGF> output_vars(y_all_adcf.size());
  
  CppAD::cg::CodeHandler<float> handler;
  handler.setVerbose(true);
  handler.makeVariables(input_vars);
  output_vars = func_final.Forward(0, input_vars);
  
  CppAD::cg::LanguageC<float> langC("float");
  CppAD::cg::LangCDefaultVariableNameGenerator<float> name_gen;
  std::ostringstream code;
  handler.generateCode(code, langC, output_vars, name_gen);
  
  std::ofstream output_file("partials.cpp");
  output_file <<
    "void computePartials(const VectorF &xk0,\n"
    "                     const VectorF &theta,\n"
    "                     VectorF &xk1,\n"
    "                     VectorF &partial_state_state,\n"
    "                     VectorF &partial_state_param)\n"
    "{\n";
  output_file << "   float x[" << input_vars.size() <<"];\n";
  output_file << "   float y[" << output_vars.size() <<"];\n";
  output_file << "   int idx = 0;\n"
    "   for(int i = 0; i < xk0.size(); i++)\n"
    "   {\n"
    "      x[i+idx] = xk0[i];\n"
    "   }\n"
    "   idx += xk0.size();\n"
    "   for(int i = 0; i < theta.size(); i++)\n"
    "   {\n"
    "      x[i+idx] = theta[i];\n"
    "   }\n"
    "   idx += theta.size();\n\n";
  output_file << code.str() << "\n";
  output_file << "   idx = 0;\n"
    "   for(int i = 0; i < xk1.size(); i++)\n"
    "   {\n"
    "      xk1[i] = y[idx+i];\n"
    "   }\n"
    "   idx += xk1.size();\n"
    "   for(int i = 0; i < partial_state_state.size(); i++)\n"
    "   {\n"
    "      partial_state_state[i] = y[idx+i];\n"
    "   }\n"
    "   idx += partial_state_state.size();\n"
    "   for(int i = 0; i < partial_state_param.size(); i++)\n"
    "   {\n"
    "      partial_state_param[i] = y[idx+i];\n"
    "   }\n"
    "   idx += partial_state_param.size();\n}";
  
  output_file.close();  
}

void Generator::generatePartialStatePrevState()
{
  std::shared_ptr<System<ADAD>> system = m_simulator->getSystem();
  
  // 1st. Use ADAD to compute an ADCF function
  VectorADAD xk0_adad(system->getStateDim());
  VectorADAD xk1_adad(system->getStateDim());
  VectorADAD theta_adad(system->getNumParams());
  
  CppAD::Independent(xk0_adad, theta_adad);
  system->setParams(theta_adad);
  m_simulator->integrate(xk0_adad, xk1_adad);
  
  CppAD::ADFun<ADCF> partial_state_prev_state_adcf(xk0_adad, xk1_adad);
  partial_state_prev_state_adcf.optimize(); //Probably will segfault.
  
  
  // 2nd. Use the ADCF function's reverse and forward to compute a CGF forward function
  VectorADCF xk0_adcf(system->getStateDim());
  VectorADCF theta_adcf(system->getNumParams());
  VectorADCF x_all_adcf(xk0_adcf.size() + theta_adcf.size());
  
  CppAD::Independent(x_all_adcf);
  for(int i = 0; i < xk0_adcf.size(); i++)
  {
    xk0_adcf[i] = x_all_adcf[i];
  }
  for(int i = 0; i < theta_adcf.size(); i++)
  {
    theta_adcf[i] = x_all_adcf[i + xk0_adcf.size()];
  }
  
  VectorADCF xk1_adcf(system->getStateDim());
  VectorADCF jacobian_adcf(system->getStateDim() * system->getStateDim()); // 
  VectorADCF y_all_adcf(xk1_adcf.size() + jacobian_adcf.size());
  
  partial_state_prev_state_adcf.new_dynamic(theta_adcf);
  xk1_adcf = partial_state_prev_state_adcf.Forward(0, xk0_adcf);
  jacobian_adcf = partial_state_prev_state_adcf.Jacobian(xk0_adcf);
  for(int i = 0; i < jacobian_adcf.size(); i++)
  {
    y_all_adcf[i] = jacobian_adcf[i];
  }
  for(int i = 0; i < xk1_adcf.size(); i++)
  {
    y_all_adcf[jacobian_adcf.size() + i] = xk1_adcf[i];  
  }
  
  CppAD::ADFun<CGF> func_final(x_all_adcf, y_all_adcf);
  func_final.optimize();
  
  // 3rd. Now to actually generate the fucking code :(  
  std::vector<CGF> input_vars(x_all_adcf.size());
  std::vector<CGF> output_vars(y_all_adcf.size());
  
  CppAD::cg::CodeHandler<float> handler;
  handler.setVerbose(true);
  handler.makeVariables(input_vars);
  output_vars = func_final.Forward(0, input_vars);
  
  CppAD::cg::LanguageC<float> langC("float");
  CppAD::cg::LangCDefaultVariableNameGenerator<float> name_gen;
  std::ostringstream code;
  handler.generateCode(code, langC, output_vars, name_gen);
  
  std::ofstream output_file("partial_state_prev_state.cpp");
  output_file << "float x[" << input_vars.size() <<"]\n";
  output_file << "float y[" << output_vars.size() <<"]\n";
  output_file << code.str();
  output_file.close();
}

void Generator::generatePartialStateParams()
{
  std::shared_ptr<System<ADAD>> system = m_simulator->getSystem();
  
  // 1st. Use ADAD to compute an ADCF function
  VectorADAD xk0_adad(system->getStateDim());
  VectorADAD xk1_adad(system->getStateDim());
  VectorADAD theta_adad(system->getNumParams());
  
  CppAD::Independent(theta_adad, xk0_adad);
  system->setParams(theta_adad);
  m_simulator->integrate(xk0_adad, xk1_adad);
  
  CppAD::ADFun<ADCF> partial_state_params_adcf(theta_adad, xk1_adad);
  partial_state_params_adcf.optimize(); //Probably will segfault.
  
  
  // 2nd. Use the ADCF function's reverse and forward to compute a CGF forward function
  VectorADCF xk0_adcf(system->getStateDim());
  VectorADCF theta_adcf(system->getNumParams());
  VectorADCF x_all_adcf(xk0_adcf.size() + theta_adcf.size());
  
  CppAD::Independent(x_all_adcf);
  for(int i = 0; i < xk0_adcf.size(); i++)
  {
    xk0_adcf[i] = x_all_adcf[i];
  }
  for(int i = 0; i < theta_adcf.size(); i++)
  {
    theta_adcf[i] = x_all_adcf[i + xk0_adcf.size()];
  }
  
  VectorADCF jacobian_adcf(system->getNumParams() * system->getStateDim());
  VectorADCF y_all_adcf(jacobian_adcf.size());
  
  partial_state_params_adcf.new_dynamic(xk0_adcf);
  jacobian_adcf = partial_state_params_adcf.Jacobian(theta_adcf);
  for(int i = 0; i < jacobian_adcf.size(); i++)
  {
    y_all_adcf[i] = jacobian_adcf[i];
  }
  
  CppAD::ADFun<CGF> func_final(x_all_adcf, y_all_adcf);
  func_final.optimize();
  
  // 3rd. Now to actually generate the fucking code :(  
  std::vector<CGF> input_vars(x_all_adcf.size()); // x0_gt.size() + x0.size()
  std::vector<CGF> output_vars(y_all_adcf.size());  // gradient size+ loss
  
  CppAD::cg::CodeHandler<float> handler;
  handler.setVerbose(true);
  handler.makeVariables(input_vars);
  output_vars = func_final.Forward(0, input_vars);
  
  CppAD::cg::LanguageC<float> langC("float");
  CppAD::cg::LangCDefaultVariableNameGenerator<float> name_gen;
  std::ostringstream code;
  handler.generateCode(code, langC, output_vars, name_gen);
  
  std::ofstream output_file("partial_state_params.cpp");
  output_file << "float x[" << input_vars.size() <<"]\n";
  output_file << "float y[" << output_vars.size() <<"]\n";
  output_file << code.str();
  output_file.close();  
}








