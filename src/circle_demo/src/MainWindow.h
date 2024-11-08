#include <memory>
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
  void drawCircle(double radius);
  void drawPixel(double x, double y, double red, double green, double blue);
  void train();
  
  
private:
  int m_width;
  int m_height;

  int m_is_ok;
  
  SDL_Window* m_window;
  unsigned int *m_pixels;

  int m_cnt = 0;
  int m_cnt_2 = 0;
  
  static constexpr int m_update_interval = 16;
  static constexpr double m_target_radius = 0.316227f;
  static constexpr double m_lr = 1e-3f;
  static constexpr double m_step_size = 1e-1f;
  static constexpr int m_num_draw_steps = 40000;
  static constexpr int m_num_train_steps = 4000;
  static constexpr int m_batch_size = 4;
  static constexpr int m_num_drawings = 4;
  
  int m_state_dim = 2;
  
  int m_num_params;
  VectorF m_params;
  VectorF m_d_squared_params;
  MatrixF m_x0;
  MatrixF m_new_x0;

  std::shared_ptr<CircleSystem<double>> m_system;
  std::shared_ptr<CircleSimulator>     m_simulator;
};
