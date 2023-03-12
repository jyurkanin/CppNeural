#include "SimulatorSimpleSystem.h"

namespace cpp_bptt
{

SimulatorSimpleSystem::SimulatorSimpleSystem(std::shared_ptr<System<float>> sys) : SimulatorF(sys) {}
SimulatorSimpleSystem::~SimulatorSimpleSystem() {}
 
void SimulatorSimpleSystem::computePartials(const VectorF &xk0,
					    const VectorF &theta,
					    const VectorF &xk1_gt,
					    VectorF &xk1,
					    VectorF &partial_state_state,
					    VectorF &partial_state_param,
					    VectorF &partial_loss_params,
					    VectorF &partial_loss_state,
					    float   &loss)
{
   float x[8];
   float y[21];
   float v[12];
   int idx = 0;
   for(int i = 0; i < xk0.size(); i++)
   {
      x[i+idx] = xk0[i];
   }
   idx += xk0.size();
   for(int i = 0; i < theta.size(); i++)
   {
      x[i+idx] = theta[i];
   }
   idx += theta.size();
   for(int i = 0; i < xk1_gt.size(); i++)
   {
      x[i+idx] = xk1_gt[i];
   }
   idx += xk1_gt.size();
   v[0] = x[2] * x[0] + x[3] * x[1];
   v[1] = x[0] + 0.05 * v[0];
   v[2] = x[4] * x[0] + x[5] * x[1];
   v[3] = x[1] + 0.05 * v[2];
   v[4] = x[2] * v[1] + x[3] * v[3];
   v[5] = x[0] + 0.05 * v[4];
   v[3] = x[4] * v[1] + x[5] * v[3];
   v[1] = x[1] + 0.05 * v[3];
   v[6] = x[2] * v[5] + x[3] * v[1];
   v[1] = x[4] * v[5] + x[5] * v[1];
   v[5] = x[1] + 0.1 * v[1];
   v[7] = x[0] + 0.1 * v[6];
   y[0] = x[0] + 0.0166667 * (v[0] + 2. * v[4] + 2. * v[6] + x[3] * v[5] + x[2] * v[7]);
   y[1] = x[1] + 0.0166667 * (v[2] + 2. * v[3] + 2. * v[1] + x[5] * v[5] + x[4] * v[7]);
   v[7] = 1 + 0.05 * x[2];
   v[5] = 0.05 * x[4];
   v[1] = x[2] * v[7] + x[3] * v[5];
   v[3] = 1 + 0.05 * v[1];
   v[5] = x[4] * v[7] + x[5] * v[5];
   v[7] = 0.05 * v[5];
   v[6] = x[2] * v[3] + x[3] * v[7];
   v[7] = x[4] * v[3] + x[5] * v[7];
   v[3] = 0.1 * v[7];
   v[4] = 1 + 0.1 * v[6];
   y[2] = 1 + 0.0166667 * (x[2] + 2. * v[1] + 2. * v[6] + x[3] * v[3] + x[2] * v[4]);
   v[6] = 0.05 * x[3];
   v[1] = 1 + 0.05 * x[5];
   v[8] = x[2] * v[6] + x[3] * v[1];
   v[9] = 0.05 * v[8];
   v[1] = x[4] * v[6] + x[5] * v[1];
   v[6] = 1 + 0.05 * v[1];
   v[10] = x[2] * v[9] + x[3] * v[6];
   v[6] = x[4] * v[9] + x[5] * v[6];
   v[9] = 1 + 0.1 * v[6];
   v[11] = 0.1 * v[10];
   y[3] = 0.0166667 * (x[3] + 2. * v[8] + 2. * v[10] + x[3] * v[9] + x[2] * v[11]);
   y[4] = 0.0166667 * (x[4] + 2. * v[5] + 2. * v[7] + x[5] * v[3] + x[4] * v[4]);
   y[5] = 1 + 0.0166667 * (x[5] + 2. * v[1] + 2. * v[6] + x[5] * v[9] + x[4] * v[11]);
   v[11] = 0.0333333 + 0.0166667 * x[2] * 0.1;
   v[0] = x[0] + 0.05 * v[0];
   v[2] = x[1] + 0.05 * v[2];
   v[9] = x[0] + 0.05 * (x[2] * v[0] + x[3] * v[2]);
   v[6] = x[1] + 0.05 * (x[4] * v[0] + x[5] * v[2]);
   v[1] = x[0] + 0.1 * (x[2] * v[9] + x[3] * v[6]);
   v[4] = 0.0166667 * x[3] * 0.1;
   v[3] = 0.0333333 + (v[4] * x[4] + v[11] * x[2]) * 0.05;
   v[7] = (v[4] * x[5] + v[11] * x[3]) * 0.05;
   v[5] = 0.0166667 + (v[7] * x[4] + v[3] * x[2]) * 0.05;
   y[6] = v[11] * v[9] + 0.0166667 * v[1] + v[3] * v[0] + v[5] * x[0];
   v[10] = x[1] + 0.1 * (x[4] * v[9] + x[5] * v[6]);
   y[7] = v[11] * v[6] + 0.0166667 * v[10] + v[3] * v[2] + v[5] * x[1];
   v[3] = (v[7] * x[5] + v[3] * x[3]) * 0.05;
   y[8] = v[7] * v[0] + v[4] * v[9] + v[3] * x[0];
   y[9] = v[7] * v[2] + v[4] * v[6] + v[3] * x[1];
   v[3] = 0.0333333 + 0.0166667 * x[5] * 0.1;
   v[7] = 0.0166667 * x[4] * 0.1;
   v[4] = (v[3] * x[4] + v[7] * x[2]) * 0.05;
   v[5] = 0.0333333 + (v[3] * x[5] + v[7] * x[3]) * 0.05;
   v[11] = (v[5] * x[4] + v[4] * x[2]) * 0.05;
   y[10] = v[4] * v[0] + v[7] * v[9] + v[11] * x[0];
   y[11] = v[4] * v[2] + v[7] * v[6] + v[11] * x[1];
   v[4] = 0.0166667 + (v[5] * x[5] + v[4] * x[3]) * 0.05;
   y[12] = v[3] * v[9] + 0.0166667 * v[1] + v[5] * v[0] + v[4] * x[0];
   y[13] = v[3] * v[6] + 0.0166667 * v[10] + v[5] * v[2] + v[4] * x[1];
   v[4] = x[6] - y[0];
   y[18] = x[2] + x[4] - v[4] - v[4];
   v[5] = x[7] - y[1];
   y[19] = x[3] + x[5] - v[5] - v[5];
   y[20] = v[4] * v[4] + v[5] * v[5] + x[3] * y[1] + x[2] * y[0] + x[5] * y[1] + x[4] * y[0];
   // variable duplicates: 4
   y[14] = y[0];
   y[15] = y[1];
   y[16] = y[0];
   y[17] = y[1];

   idx = 0;
   for(int i = 0; i < xk1.size(); i++)
   {
      xk1[i] = y[idx+i];
   }
   idx += xk1.size();
   for(int i = 0; i < partial_state_state.size(); i++)
   {
      partial_state_state[i] = y[idx+i];
   }
   idx += partial_state_state.size();
   for(int i = 0; i < partial_state_param.size(); i++)
   {
      partial_state_param[i] = y[idx+i];
   }
   idx += partial_state_param.size();
   for(int i = 0; i < partial_loss_params.size(); i++)
   {
      partial_loss_params[i] = y[idx+i];
   }
   idx += partial_loss_params.size();
   for(int i = 0; i < partial_loss_state.size(); i++)
   {
      partial_loss_state[i] = y[idx+i];
   }
   idx += partial_loss_state.size();
   loss = y[idx];
}

} //cpp_bptt
