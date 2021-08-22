# Algorithmic Differentiation of Dynamic Systems
This library provides tools for:
- Defining and composing linear dynamic systems with symbolic variables which can be automatically differentiated
- Optimizing resulting systems wrt 
  - [x] H2 norm
  - [ ] Hinf norm
- Constraining the optimization problem to maintain
  - [x] Stability (via Routh Table)
  - [x] Passivity (numerically)
  
---
## What's different from Matlab / Control System Toolbox / etc
The objectives and constraints here can be algorithmically differentiated wrt the symbolic variables, allowing the automatic calculation of Jacobian and Hessian matrices, making the constrained optimization efficient.

This is different from H2 or Hinf design in that it is more flexible.  It can accomodate structured optimization, where only certain parameters (e.g. of a hierarchical controller, dynamic parameters) can be optimized. It can also accomodate arbitrary (differentiable) constraints on the system, such as stability. As multiple constraints can be applied, stabilty can be enforced for several different system conditions.

---
## How does it work?
The *H2 norm* can be calculated via the Controllability/Observability Gramian, which can in turn be calculated from the solution to a Lyapunov equation.
For continuous, SISO, LTI systems this equation looks like AX+XA' = Q. This equation is differentiable - see, e.g. [here](https://arxiv.org/abs/2011.11430). The second-order derivative (i.e. Hessian matrix) can be derived without too much headache, as the first-order derivative is also a Lyapunov equation.
Doing the naive solve (vectorize the equation then linear solve) works, but isn't stable for poorly-conditioned systems.  The Bartels-Stewart algorithm for solving Lyapuonov equations is more robust (it's what Matlab uses), the one interfaced by Scipy is used here for ease of install.


*Stability* is currently via Routh Tables which can be calculated algorithmically and are differentiable -- recall that the system is stable if there's no sign changes in the first column. Denote the elements in this first column ri, we can add the constraint r0*ri > 0 for all i =/= 0 to our constraints.

*Passivity* can be enforced by separating the real/imaginary components of a transfer function (multiply by complex conjugate so denominator is real, then split even/odd powers of numerator).  The constraint Re{G}>0 can then be evaluated at a range of frequencies and added to the constraints.



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

[CasADi](https://web.casadi.org/) provides the base autodiff functionality and a nice interface for NLP solvers. IPOPT should come with your CasADi install, which has worked well on the test problems so far.


---
## Dev Plans:
- [x] 1st and 2nd derivative from Lyap
- [ ] Riccati equation + 1st and 2nd derivative
- [ ] Hinf norm. Some work already in [sensitivity](https://arc.aiaa.org/doi/pdf/10.2514/3.21138)
