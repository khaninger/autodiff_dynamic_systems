# Description:   Integrate scipy solver for the Lyapunov equation in Casadi
# Contributor:   Kevin Haninger {khaninger@gmail.com}
# First written: 2021.08.30

# Library imports
import casadi as ca
import numpy as np

class ForwardFun(ca.Callback):
    def __init__(self, d, nfwd, name, inames, onames, opts):
        ca.Callback.__init__(self)
        self.d = d
        self.nfwd = nfwd
        self.fwd_hess = []
        self.construct(name, opts)
    def get_n_in(self):  return 9
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        if i < 3:
            return ca.Sparsity.dense(self.d, self.d)
        else:
            return ca.Sparsity.dense(self.d, self.d* self.nfwd)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(self.d, self.d*self.nfwd)

    def eval(self, arg):
        # The argument order seems to be: args to f(), solution to f(), then the deriv of inputs to f().
        from scipy.linalg import solve_continuous_lyapunov # Sorry, but the Casadi VM cant find if we import above
        A = arg[0]
        Q = arg[1]
        P = arg[2]
        ret = []
        for i in range(self.nfwd):
            Ad = arg[3][:,self.d*i:self.d*(i+1)]
            Qd = arg[4][:,self.d*i:self.d*(i+1)]
            Pd = solve_continuous_lyapunov(A, -(Ad @ P + P @ Ad.T + Qd))
            ret.append(Pd)
        return [np.hstack(ret)]
    def has_forward(self, nfwd): return True
    def get_forward(self, nfwd, name, inames, onames, opts):
        self.fwd_hess.append(ForwardForwardFn(self.d, nfwd, name, inames, onames, opts))
        return self.fwd_hess[-1]

class ForwardForwardFn(ca.Callback):
    def __init__(self, d, nfwd, name, inames, onames, opts):
        ca.Callback.__init__(self)
        self.d = d
        self.nfwd = nfwd
        self.construct(name, opts)
    def get_n_in(self):  return 11
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        if i < 6:
            return ca.Sparsity.dense(self.d, self.d)
        else:
            return ca.Sparsity.dense(self.d, self.d*self.nfwd)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(self.d, self.d*self.nfwd)
    def eval(self, arg):
        from scipy.linalg import solve_continuous_lyapunov # Sorry, but the Casadi VM cant find if we import above
        # First 5 arguments are the inputs to fwd_lyap (i.e. the points about which Pd is calculated)
        A = arg[0]
        P = arg[2]
        Ad = arg[3]
        # Pd is the solution to the first fwd_lyap
        Pd = arg[5]
        ret = []
        for i in range(self.nfwd):
            # These are tangent inputs to the individual arguments
            Ad_ = arg[6][:,self.d*i:self.d*(i+1)]
            Pd_ = arg[8][:,self.d*i:self.d*(i+1)]
            Add_ = arg[9][:,self.d*i:self.d*(i+1)]
            Qdd_ = arg[10][:,self.d*i:self.d*(i+1)]
            Q__ = (Ad_ @ Pd + Pd @ Ad_.T) + (Add_ @ P + Ad @ Pd_ + Pd_ @ Ad.T + P @ Add_.T) + Qdd_
            ret.append(solve_continuous_lyapunov(A,-Q__))
        return [np.hstack(ret)]


class RiccatiSolver(ca.Callback):
    '''
    Class for lyapunovRiccati solver which uses the scipy.linalg solver,
    and provides the forward/reverse gradient explicitly.

    This class largely written following Casadi example code
    /test/python/function.py, but the devs have warned this
    interface is not a priority/stable.
    '''
    from scipy.linalg import solve_discrete_are
    def __init__(self, name, m, n, opts = {}):
        ca.Callback.__init__(self)
        self.m = m
        self.n = n
        self.cb_fwd = []
        self.construct(name, opts)

    def get_n_in(self):  return 4
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        if i == 0 or i == 2: return ca.Sparsity.dense(self.m, self.m)
        if i == 1: return ca.Sparsity.dense(self.m, self.n)
        if i == 3: return ca.Sparsity.dense(self.n, self.n)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(self.m, self.m)

    def has_forward(self, nfwd): return False
    def has_reverse(self, nadj): return (nadj==1) # Adjoint doesn't currently support parallel eval (just 1 eval)

    def eval(self, arg):
        from scipy.linalg import solve_discrete_are
        A = arg[0]
        B = arg[1]
        Q = arg[2]
        R = arg[3]
        return [solve_discrete_are(A, B, Q, R)]

    def get_forward(self, nfwd, name, inames, onames, opts):
        self.cb_fwd.append(ForwardFun(self.m, nfwd, name, inames, onames, opts))
        return self.cb_fwd[-1] # This object needs to be kept alive for later evaluation

    def get_reverse(self, nadj, name, inames, onames, opts):
        assert(nadj==1)
        class BackwardFun(ca.Callback):
            def __init__(self, m, n):
                ca.Callback.__init__(self)
                self.m = m
                self.n = n
                self.construct(name, opts)
            def get_n_in(self):  return 6
            def get_n_out(self): return 4
            def get_sparsity_in(self, i):
                if i == 0 or i == 2 or i == 4 or i == 5: return ca.Sparsity.dense(self.m, self.m)
                if i == 1: return ca.Sparsity.dense(self.m, self.n)
                if i == 3: return ca.Sparsity.dense(self.n, self.n)
                
            def get_sparsity_out(self, i): 
                if i == 0 or i == 2: return ca.Sparsity.dense(self.m, self.m)
                if i == 1: return ca.Sparsity.dense(self.m, self.n)
                if i == 3: return ca.Sparsity.dense(self.n, self.n)

            def eval(self, arg):
                # The argument order seems to be: args to f(), solution to f(), then the deriv of inputs to f().
                from scipy.linalg import solve_discrete_lyapunov # Sorry, but the Casadi VM cant find if we import above
                A = arg[0]
                B = arg[1]
                Q = arg[2]
                R = arg[3]
                P = arg[4]
                Pbar = arg[5]

                K = ca.inv(R+B.T@P@B)@B.T@P@A
                Atilde = A-B@K
                S = solve_discrete_lyapunov(Atilde, 0.5*(Pbar.T+Pbar))
                return [2*P@Atilde@S, -2*P@Atilde@S@(K.T), S, K@S@(K.T)]

        self.cb_rev = BackwardFun(self.m, self.n)
        return self.cb_rev # This object needs to be kept alive for later evaluation
