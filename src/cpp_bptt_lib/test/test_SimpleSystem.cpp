#include "SimpleSystem.h"
#include "SimulatorSimpleSystem.h"
#include "cpp_bptt.h"

#include <memory>
#include <iostream>

#include "gtest/gtest.h"

namespace {
  
  class SimpleSystemFixture : public ::testing::Test
  {
  public:
    std::vector<VectorF> m_gt_list;
    VectorF m_x0;
    VectorF m_params;
    
    std::shared_ptr<SimpleSystem<ADF>>     m_system_adf;
    std::shared_ptr<SimulatorAD>           m_simulator_adf;
    std::shared_ptr<SimpleSystem<float>>   m_system_f;
    std::shared_ptr<SimulatorSimpleSystem> m_simulator_f;
    
    SimpleSystemFixture()
    {
      m_system_adf = std::make_shared<SimpleSystem<ADF>>();
      m_simulator_adf = std::make_shared<SimulatorAD>(m_system_adf);
      
      m_system_f = std::make_shared<SimpleSystem<float>>();
      m_simulator_f = std::make_shared<SimulatorSimpleSystem>(m_system_f);

      m_params = VectorF::Random(m_system_adf->getNumParams());
      m_x0 = VectorF::Random(m_system_adf->getStateDim());
      m_gt_list.resize(m_system_adf->getNumSteps());

      for(int i = 0; i < m_gt_list.size(); i++)
      {
	m_gt_list[i] = VectorF::Random(m_system_f->getStateDim());
      }
      
      std::cout << "constructor\n";
    }
    ~SimpleSystemFixture(){}
    
    VectorF getGradientSimple()
    {
      VectorAD x0(m_system_adf->getStateDim());
      for(int i = 0; i < x0.size(); i++)
      {
	x0[i] = m_x0[i];
      }
      
      VectorAD params(m_system_adf->getNumParams());
      for(int i = 0; i < params.size(); i++)
      {
	params[i] = m_params[i];
      }
      
      CppAD::Independent(params);
      m_system_adf->setParams(params);
      
      std::vector<VectorAD> x_list(m_system_adf->getNumSteps());
      m_simulator_adf->forward(x0, x_list);
      
      VectorAD loss(1);
      
      loss[0] = 0;
      for(int i = 0; i < x_list.size(); i++)
	{
	  for(int j = 0; j < m_system_adf->getStateDim(); j++)
	    {
	      ADF err = x_list[i][j] - m_gt_list[i][j];
	      loss[0] += err*err;
	    }
	}
      
      CppAD::ADFun<float> func(params, loss);
      
      VectorF y0(1);
      y0[0] = 1;
      
      return func.Reverse(1, y0);
    }
    
    VectorF getGradientBPTT()
    {
      VectorAD params(m_system_adf->getNumParams());
      for(int i = 0; i < params.size(); i++)
      {
	params[i] = m_params[i];
      }
      m_system_adf->setParams(params);
      
      VectorF gradient;
      float loss;
            
      VectorF x0_f(m_system_adf->getStateDim());
      for(int i = 0; i < x0_f.size(); i++)
      {
	x0_f[i] = m_x0[i];
      }
      
      m_simulator_adf->forward_backward(x0_f, m_gt_list, gradient, loss);
      
      return gradient;
    }
    
    VectorF getGradientHard()
    {
      VectorF x0(m_system_f->getStateDim());
      for(int i = 0; i < x0.size(); i++)
	{
	  x0[i] = m_x0[i];
	}
      
      VectorF params(m_system_f->getNumParams());
      for(int i = 0; i < params.size(); i++)
      {
	params[i] = m_params[i];
      }
            
      m_system_f->setParams(params);
      
      VectorF gradient;
      float loss;
      
      m_simulator_f->forward_backward(x0, m_gt_list, gradient, loss);
      
      return gradient;
    }
    
  };
    
  TEST_F(SimpleSystemFixture, validate_gradient_easy)
  {
    VectorF grad_simple = getGradientSimple();
    VectorF grad_bptt = getGradientBPTT();

    for(int i = 0; i < m_system_f->getNumParams(); i++)
    {
      std::cout << grad_simple[i] << ", " << grad_bptt[i] << "\n";
      EXPECT_LE(fabs(grad_simple[i] - grad_bptt[i]), fabs(1e-4f*grad_simple[i]));
    }
  }

  TEST_F(SimpleSystemFixture, validate_gradient_hard)
  {
    VectorF grad_simple = getGradientSimple();
    VectorF grad_hard = getGradientHard();

    for(int i = 0; i < m_system_f->getNumParams(); i++)
    {
      std::cout << grad_simple[i] << ", " << grad_hard[i] << "\n";
      EXPECT_LE(fabs(grad_simple[i] - grad_hard[i]), fabs(1e-4f*grad_simple[i]));
    }
  }

  
}
