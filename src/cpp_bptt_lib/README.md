Assuming an ode of the form $\dot{x} = f(x, \theta)$
Where f is any continuous function with parameters \theta. f could be a neural net or a model that includes a neural net.
And then using a Runge-Kutta 4th order method to integrate: $x_{k+1} = g(x_k, \theta)$
Collect all $x_k$ into a big vector $X$
Assuming loss function $L(X, \theta) = L(x_0, \theta) + L(x_1, \theta) + \ldots$
We want to compute $\frac{dL}{d\theta}$.
We are going to need these partials/jacobians:
$\frac{\partial L}{\partial x_k}$ //
$\frac{dL}{d\theta}$ //
$\frac{\partial g}{\partial \theta}$ //
$\frac{\partial g}{\partial x_{k-1}}$ //


