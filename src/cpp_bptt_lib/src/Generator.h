#include <types/Tensors.h>
#include <types/Scalars.h>
#include "SimulatorADAD.h"
#include <memory>

// Generates something.
class Generator
{
public:
  Generator() = default;
  ~Generator() = default;

  //I could have used a function ptr, but I hate those. This is much nicer to look at.
  void setSimulator(const std::shared_ptr<SimulatorADAD> &simulator){ m_simulator = simulator; }
  void initialize();
  
private:
  void generatePartials();              // Partials of state wrt prev state and state wrt params
  void generatePartialLossState();      // Partial of Loss wrt the current state
  void generatePartialStatePrevState(); // Partial of state wrt the previous state
  void generatePartialStateParams();    // Partial of state wrt the params

  std::shared_ptr<SimulatorADAD> m_simulator;
};
