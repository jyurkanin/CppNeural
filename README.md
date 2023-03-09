I made a library for training neural networks in C++
Also I made a library for backpropagation through time for optimizing neural ODE's with auto-generated code.
The generated code has no unrolled loops which makes compilation much faster. With huge unrolled loops, compiling can take like 20mins
because the generated code is like 100kloc.
No adjoint method (cringe)
A cool demo
Good luck

