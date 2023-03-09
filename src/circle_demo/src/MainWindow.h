#include <SDL2/SDL.h>
#include "CircleSystem.h"
#include "CircleSimulator.h"


class MainWindow
{
public:
  MainWindow();
  ~MainWindow();

  int init();
  int ok();
  int del();
  void loop();
  void drawODEs();
  void drawCircle(float radius);
  void drawPixel(float x, float y, float red, float green, float blue);
  void train();
  
  
private:
  int m_width;
  int m_height;

  int m_is_ok;
  
  SDL_Window* m_window;
  unsigned int *m_pixels;

  int m_cnt = 0;
  
  static constexpr int m_update_interval = 64;
  static constexpr float m_target_radius = 0.1f;
  static constexpr float m_lr = 1e-3f;
  static constexpr float m_step_size = 1e-1f;
  static constexpr int m_num_draw_steps = 2048;
  static constexpr int m_num_train_steps = 2048;
  static constexpr int m_batch_size = 4;
  static constexpr int m_num_drawings = 4;
  static constexpr int m_num_hidden_nodes = 8;
  
  int m_state_dim = 2;
  
  int m_num_params;
  VectorF m_params;
  VectorF m_d_squared_params;
  MatrixAD m_x0;
  MatrixAD m_new_x0;

  CircleSystem<ADF> m_system;
};
