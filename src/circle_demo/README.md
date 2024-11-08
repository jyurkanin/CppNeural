This uses a neural ODE and it optimizes this neural ODE so that it becomes a limit cycle.
Basically it does this by giving it a penalty for when the norm of the state is not
equal to a target radius.

