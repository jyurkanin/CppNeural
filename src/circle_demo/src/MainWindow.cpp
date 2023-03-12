#include "MainWindow.h"

#include <iostream>
#include <stdlib.h>


MainWindow::MainWindow()
{
  int argc = 0;
  char **argv = 0;

  m_system = std::make_shared<CircleSystem<float>>();
  m_system->setNumSteps(m_num_train_steps);
  m_system->setTimestep(.1);
  m_system->setLearningRate(m_lr);
  
  m_simulator = std::make_shared<CircleSimulator>(m_system);
  
  m_width = 1024;
  m_height = 1024;
  
  m_num_params = m_system->getNumParams();
  m_params = -.01*VectorF::Random(m_num_params, 1);
  m_d_squared_params = VectorF::Zero(m_num_params, 1);
  
  m_state_dim = m_system->getStateDim();
  
  m_x0 = 2*MatrixF::Random(m_state_dim, m_batch_size) - (MatrixF::Ones(m_state_dim, m_batch_size));

  float starting_radius = .1;
  for(int i = 0; i < m_x0.cols(); i++)
  {
    float norm = (m_x0(0,i)*m_x0(0,i)) + (m_x0(1,i)*m_x0(1,i));
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
  VectorF x0(m_system->getStateDim());
  std::vector<VectorF> gt_list(m_system->getNumSteps());
  VectorF d_params(m_system->getNumParams());
  VectorF d_params_sum = VectorF::Zero(m_system->getNumParams());
  float loss;
  
  for(int i = 0; i < gt_list.size(); i++)
  {
    gt_list[i] = VectorF::Zero(m_system->getStateDim());
  }

  m_simulator->setParams(m_params);
  
  for(int j = 0; j < m_batch_size; j++)
  {
    if(j == 0) // draw 1 the same.
    {
      for(int k = 0; k < m_new_x0.rows(); k++)
      {
	x0[k] = m_new_x0(k,0);
      }
    }
    else // draw the others randomly
    {
      x0 = VectorF::Random(m_state_dim);
    }
    
    m_simulator->forward_backward(x0, gt_list, d_params, loss);
    d_params_sum += d_params;
  }

  VectorF d_params_avg = d_params_sum / m_batch_size;
  
  std::cout << "Score " << loss << "\n";
  
  // RMS Grad. Works way better than gradient descent.
  for(int i = 0; i < m_params.size(); i++)
  {
    m_d_squared_params[i] = (.9*m_d_squared_params[i]) + (.1*d_params_avg[i]*d_params_avg[i]);
  }
  
  for(int i = 0; i < m_params.size(); i++)
  {
    m_params[i] -= (m_lr / (sqrtf(m_d_squared_params[i]) + 1e-6f)) * d_params_avg[i];
  }
  
}

void MainWindow::drawODEs()
{
  VectorF x_k1(m_state_dim);
  VectorF x_k(m_state_dim);
  
  m_simulator->setParams(m_params);
  for(int j = 0; j < m_batch_size; j++)
  {

    // set initial state
    if(j == 0) // draw 1 the same.
    {
      for(int k = 0; k < m_new_x0.rows(); k++)
      {
	x_k[k] = m_new_x0(k,0);
      }
    }
    else // draw the others randomly
    {
      x_k = VectorF::Random(m_state_dim);
    }
    
    for(int i = 0; i < m_num_draw_steps; i++)
    {
      m_simulator->integrate(x_k, x_k1);

      if(m_cnt == m_update_interval && j == 0 && i == 16)
      {
	m_cnt = 0;
	for(int m = 0; m < m_new_x0.rows(); m++)
	{
	  m_new_x0(m,0) = x_k[m];
	}

      }
      
      float red = ((float)j / m_batch_size);
      float green = 1 - red;
      float blue = 0;
      drawPixel(x_k1[0], x_k1[1], red,green,blue);

      x_k = x_k1;
    }
  }
  
  m_cnt++;
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

  

  if(m_cnt_2 == 32)
  {
    m_cnt_2 = 0;
    SDL_FillRect(window_surface, NULL, 0);
    drawODEs();
    drawCircle(m_target_radius);
    drawPixel(-m_target_radius, -m_target_radius, 1,1,1);
    drawPixel(m_target_radius, m_target_radius, 1,1,1);
  
    SDL_UpdateWindowSurface(m_window);
  }
  m_cnt_2++;
}
