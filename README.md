# Algorithmic Differentiation of Dynamic Systems
This library provides tools for:
- Building and composing linear dynamic systems with symbolic variables to represent parameters or design variables.
- Optimizing these systems over the design variables
  - [x] H2 norm
- Constraining the optimization problem to maintain system properties
  - [x] Stability (via Routh Table)
  - [x] Passivity
  
---
## What's different from Matlab / Control System Toolbox / etc
The objectives and constraints here can be algorithmically differentiated wrt the symbolic variables, allowing the automatic calculation of Jacobian and Hessian matrices, making the constrained optimization efficient.

---
## How's it work?
The *H2 norm* can be calculated via the Controllability/Observability Gramian, which can in turn be calculated from the solution to a Lyapunov equation.
For continuous, SISO, LTI systems this equation looks like AX+XA' = Q. This equation is differentiable - see, e.g. [here](https://arxiv.org/abs/2011.11430) first-order second-order can be derived without too much headache.
Doing the naive solve (vectorize the equation then linear solve) works, but isn't stable for poorly-conditioned systems.  The Bartels-Stewart algorithm is a bit more robust, we use the one interfaced by Scipy for ease of install.


*Stability* is currently via Routh Tables which can be calculated algorithmically and are differentiable -- recall that the system is stable if there's no sign changes in the first column. Denote the elements in this first column ri, we can add the constraint r0ri > 0 to our optimization problem.

*Passivity* can be enforced by separating the real/imaginary components of a transfer function (multiply by complex conjugate so denominator is real, then split even/odd powers of numerator).  The constraint Re{G}>0 can then be added to the optimization problem.

---
## What's here?
- autodiff_sys: a system with symbolic parameters in the dynamics
- helper_fns: tools for manipulating systems, time-delay approximations, etc
- lyap_solver: implements a twice-differentiable solver to Lyapunov equations
- admittance_gains_calc: notebook with a bit more involved example of how it can be used

---
## How can I use it?
This is tested with:
- Python 3.5.0
- Numpy 1.19.5
- Casadi 3.5.0

[CasADi](https://web.casadi.org/) provides the base autodiff functionality and a nice interface for NLP solvers. IPOPT should come with your CasADi install, which has been working great for me so far.


---
## Dev Plans:
- [x] 1st and 2nd derivative from Lyap
- [ ] Riccati equation + 1st and 2nd derivative
- [ ] Check if eigenvalues via the QR algorithm computationally friendly
- [ ] Hinf norm. Some work already in [sensitivity](https://arc.aiaa.org/doi/pdf/10.2514/3.21138)
- [ ] Discrete-time systems
