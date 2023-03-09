#include "MainWindow.h"

#include <iostream>
#include <stdlib.h>


MainWindow::MainWindow()
{
  int argc = 0;
  char **argv = 0;
  
  m_width = 1024;
  m_height = 1024;

  m_system->setNumSteps(1000);
  m_system->setTimestep(.01);
  m_system->setLearningRate(.001);
  
  m_num_params = m_system.getNumParams();
  m_params = -.01*VectorF::Random(m_num_params, 1);
  m_d_squared_params = VectorF::Zero(m_num_params, 1);
  
  m_state_dim = m_system.getStateDim();
  
  m_x0 = 2*MatrixAD::Random(m_state_dim, m_batch_size) - (MatrixAD::Ones(m_state_dim, m_batch_size));

  float starting_radius = .1;
  for(int i = 0; i < m_x0.cols(); i++)
  {
    ADF norm = CppAD::sqrt((m_x0(0,i)*m_x0(0,i)) + (m_x0(1,i)*m_x0(1,i)));
    m_x0(0,i) = starting_radius*m_x0(0,i)/norm;
    m_x0(1,i) = starting_radius*m_x0(1,i)/norm;
  }
  
  m_new_x0 = m_x0;
}

MainWindow::~MainWindow()
{
  
}

int MainWindow::init()
{
  if(SDL_Init( SDL_INIT_VIDEO ) < 0){
    printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
    SDL_DestroyWindow( m_window );
    SDL_Quit();
    return 1;
  }
  
  m_window = SDL_CreateWindow( "Idk", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, m_width, m_height, SDL_WINDOW_SHOWN );
  if(m_window == NULL) {
    printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
    SDL_DestroyWindow( m_window );
    SDL_Quit();
    return 1;
  }

  SDL_Surface *window_surface = SDL_GetWindowSurface(m_window);
  m_pixels = (unsigned int *) window_surface->pixels;
  int width = window_surface->w;
  int height = window_surface->h;
  unsigned num_pixels = width*height;
  printf("Pixel format: %s\n", SDL_GetPixelFormatName(window_surface->format->format));  
  printf("This shit should match %d %d\t%d %d\n", width, height, m_width, m_height);

  m_is_ok = 1;
  
  return 0;
}

int MainWindow::ok()
{
  return m_is_ok;
}

int MainWindow::del()
{
  return 0;
}

void MainWindow::train()
{ 
  VectorAD ad_params(m_params.size());
  for(int i = 0; i < m_params.size(); i++)
  {
    ad_params[i] = m_params[i];
  }
  
  CppAD::Independent(ad_params);
  
  // Load params into the ffwd model
  m_system.setParams(ad_params);
  
  MatrixAD x_k = m_new_x0; //4*MatrixAD::Random(m_state_dim, m_batch_size) - (2*MatrixAD::Ones(m_state_dim, m_batch_size));
  MatrixAD x_k1(m_state_dim,m_batch_size);
  MatrixAD xd(m_state_dim,m_batch_size);
  
  ADF score = 0;
  ADF diff_sum = 0;
  ADF radius_sum = 0;
  for(int j = 0; j < m_num_train_steps; j++)
  {
    m_system.forward(x_k, xd);
    
    x_k1 = x_k + xd*m_step_size;

    if(j == 16){
      if (m_cnt == m_update_interval) {
        // m_new_x0(0,0) = x_k(0,0);
        // m_new_x0(1,0) = x_k(1,0);
        m_new_x0 = x_k;
        m_cnt = 0;
      }
    }

    for(int i = 0; i < x_k1.cols(); i++)
    {
      ADF radius = CppAD::sqrt(x_k1(0,i)*x_k1(0,i) + x_k1(1,i)*x_k1(1,i));
      ADF dx = m_target_radius - x_k1(0,i);
      ADF dy = m_target_radius - x_k1(1,i);

      ADF dr = m_target_radius - radius;
      radius_sum += dr*dr;
      // radius_sum += dx*dx + dy*dy;
      
      /* dx = -m_target_radius - x_k1(0,i); */
      /* dy = -m_target_radius - x_k1(1,i); */
      /* radius_sum += dx*dx + dy*dy; */
      
      //ADF norm = CppAD::sqrt(xd(0,i)*xd(0,i) + xd(1,i)*xd(1,i));
      //diff_sum += CppAD::exp(-10*norm);

      ADF norm = CppAD::exp(-10*(CppAD::abs(xd(0,i)) + CppAD::abs(xd(1,i))));
      diff_sum += norm;
    }
    
    x_k = x_k1;
  }
  
  score = radius_sum + 1e-1f*diff_sum;
  
  VectorAD y(1);
  y[0] = score / (float) m_batch_size;
  CppAD::ADFun<float> func(ad_params, y);

  std::cout << "Score " << CppAD::Value(score) << ", " << CppAD::Value(radius_sum) << ", " << CppAD::Value(diff_sum) << "\n";
  
  VectorF y0(1);
  y0[0] = 1;
  VectorF d_params = func.Reverse(1, y0);
  
  // RMS Grad. Works way better than gradient descent.
  for(int i = 0; i < m_params.size(); i++)
  {
    m_d_squared_params[i] = (.9*m_d_squared_params[i]) + (.1*d_params[i]*d_params[i]);
  }
  
  for(int i = 0; i < m_params.size(); i++)
  {
    m_params[i] -= (m_lr / (sqrtf(m_d_squared_params[i]) + 1e-6f)) * d_params[i];
  }
  
  m_cnt++;
}

void MainWindow::drawODEs()
{
  
  MatrixAD x_k = MatrixF::Random(m_state_dim,m_num_drawings);
  MatrixAD x_k1(m_state_dim,m_num_drawings);
  MatrixAD xd(m_state_dim,m_num_drawings);

  for(int i = 0; i < m_new_x0.rows(); i++)
  {
    x_k(i,0) = CppAD::Value(m_new_x0(i,0));
  }
  
  for(int i = 0; i < m_num_draw_steps; i++)
  {
    m_system.forward(x_k, xd);
    
    x_k = x_k + xd*m_step_size;

    for(int j = 0; j < x_k.cols(); j++)
    {
      float red = ((float)j / x_k.cols());
      float green = 1 - red;
      float blue = 0;
      
      drawPixel(CppAD::Value(x_k(0,j)), CppAD::Value(x_k(1,j)), red,green,blue);
    }
  }
}

void MainWindow::drawCircle(float radius)
{
  for(float i = 0; i < 7; i+=.01)
  {
    drawPixel(radius*cosf(i), radius*sinf(i), 0,0,1);
  }
}

void MainWindow::drawPixel(float x, float y, float red, float green, float blue)
{
  const float scale = 300;
  int row = std::max(std::min((int) (y*scale) + (m_height/2), m_height), 0);
  int col = std::max(std::min((int) (x*scale) + (m_width/2), m_width), 0);

  unsigned r = (red*0xFF);
  unsigned g = (green*0xFF);
  unsigned b = (blue*0xFF);
  
  r = (r & 0xFF) << 16;
  g = (g & 0xFF) << 8;
  b = (b & 0xFF);
  
  m_pixels[(m_width*row) + col] = r | g | b;
}

void MainWindow::loop()
{
  SDL_Surface *window_surface = SDL_GetWindowSurface(m_window);
  
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT){
      SDL_DestroyWindow( m_window );
      SDL_Quit();

      m_is_ok = 0;
      return;
    }
	
    if (event.type == SDL_WINDOWEVENT) {
      if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
	m_pixels = (unsigned int *) window_surface->pixels;
	int width = window_surface->w;
	int height = window_surface->h;
	printf("Size changed: %d, %d\n", width, height);
      }
    }
  }

  train();

  SDL_FillRect(window_surface, NULL, 0);
  drawODEs();
  drawCircle(m_target_radius);
  drawPixel(-m_target_radius, -m_target_radius, 1,1,1);
  drawPixel(m_target_radius, m_target_radius, 1,1,1);
  
  SDL_UpdateWindowSurface(m_window);
}
