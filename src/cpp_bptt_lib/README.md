Assuming an ode of the form $\dot{x} = f(x, \theta)$
Where f is any continuous function with parameters \theta. f could be a neural net or a model that includes a neural net.
And then using a Runge-Kutta 4th order method to integrate: $x_{k+1} = g(x_k, \theta)$
Collect all $x_k$ into a big vector $X$
Assuming loss function $L(X) = L(x_0) + L(x_1) + \ldots$
We want to compute $\frac{dL}{d\theta}$.
We are going to need these partials:
$\frac{\partial L}{\partial x_k}$
$\frac{\partial g}{\partial \theta}$
$\frac{\partial g}{\partial x_{k-1}}$

$\frac{dx_k}{d\partial} = \frac{\partial g}{\partial x_{k-1}}*\frac{dx_{k-1}}{d\theta} + \frac{\partial g}{ \partial \theta}$



Okay, so there is going to be a Generator class which generates the source code for the gradients
Simulator abstractbase class with 2 derived classes.
An CppAD Simulator class and a CppADCodeGen'd Simulator class.

