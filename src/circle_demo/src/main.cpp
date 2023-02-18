#include <vector>
#include <memory>


#include "MainWindow.h"


int main(int argc, char **argv)
{
  MainWindow window;

  if(window.init())
  {
    printf("Failed to initialize Main Window\n");
  }
  
  while(window.ok())
  {
    window.loop();
    usleep(1000);
  }
  
  if(window.del())
  {
    printf("Deleting the Main Window failed\n");
  }

}
