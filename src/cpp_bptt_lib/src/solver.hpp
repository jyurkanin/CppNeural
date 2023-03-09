#include <Eigen/Dense>

template<typename Scalar>
void ODE(const Eigen::Matrix<Scalar,Eigen::Dynamic,1> &X,
	       Eigen::Matrix<Scalar,Eigen::Dynamic,1> &Xd,
	 const Eigen::Matrix<Scalar,Eigen::Dynamic,1> &theta)
{
  Xd[0] = theta[0]*X[1] + theta[4]*X[0];
  Xd[1] = -X[0] + theta[1]*X[1] - (theta[2]*X[1]*X[0]*X[0]) + (theta[3]*X[0]*X[0]*X[0]);
}

template<typename Scalar>
void rk4_step(Eigen::Matrix<Scalar,Eigen::Dynamic,1> &X, const Eigen::Matrix<Scalar,Eigen::Dynamic,1> &theta){
  Scalar ts(.01f);
  Scalar ts_6(.01f/6.0f);
  Scalar c1(1);
  Scalar c2(2);
  Scalar c3(2);
  Scalar c4(1);
  
  int state_dim = X.size();
  
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> k1(state_dim);
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> k2(state_dim);
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> k3(state_dim);
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> k4(state_dim);
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> temp(state_dim);
  
  ODE(X, k1, theta);
  temp = X + .5*ts*k1;
  
  ODE(temp, k2, theta);
  temp = X + .5*ts*k2;
  
  ODE(temp, k3, theta);
  temp = X + ts*k3;
  
  ODE(temp, k4, theta);
  X += (ts_6)*(k1 + c2*k2 + c3*k3 + k4);
}

template<typename Scalar>
void solve_loss(const Eigen::Matrix<Scalar,Eigen::Dynamic,1> &X0, const Eigen::Matrix<Scalar,Eigen::Dynamic,1> &theta, Eigen::Matrix<Scalar,Eigen::Dynamic,1> &loss, int total_steps){  
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> X = X0;
  
  Scalar desired_radius(4.0f); //
  Scalar radius;
  Scalar temp;
  loss[0] = 0;
  
  for(int i = 0; i < total_steps; i++){
    rk4_step(X, theta);
    
    radius = CppAD::sqrt(X[0]*X[0] + X[1]*X[1])*theta[5];
    temp = radius - desired_radius;
    loss[0] += temp*temp; // + (.01f / (.0001f + radius*radius));
  }
  

}
