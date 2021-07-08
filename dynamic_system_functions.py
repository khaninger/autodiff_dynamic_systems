import casadi as ca
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class sys():
    def __init__(self, num, den):
        # Check if we can cancel some poles/zeros at 0
        self.num = num
        self.den = den

    def __mul__(self, other):
        new_num = con(self.num, other.num)
        new_den = con(self.den, other.den)
        return sys(new_num, new_den)

    def __rmul__(self, other):
        new_num = con(self.num, other.num)
        new_den = con(self.den, other.den)
        return sys(new_num, new_den)

    def __truediv__(self, other):
        new_num = con(self.num, other.den)
        new_den = con(self.den, other.num)
        return sys(new_num, new_den)

    def __add__(self, other):
        if isinstance(other, sys):
            new_num = ladd(con(self.num, other.den), con(self.den, other.num))
            new_den = con(self.den, other.den)
        else:
            new_num = ladd(other*self.den, self.num)
            new_den = self.den
        return sys(new_num, new_den)

    def __str__(self):
        return 'num: ' + str(self.num) + ' den: ' + str(self.den)

    def h2_norm(self, sol = 'lapacklu'):
        while ca.MX.is_zero(self.num[-1]) and ca.MX.is_zero(self.den[-1]):
            print('Cancelling pole/zero at 0')
            self.num.pop()
            self.den.pop() 
        if ca.MX.is_zero(self.den[-1]):
            print("System has pole at zero; H2 undefined")
        if len(self.num) >= len(self.den):
            print("System has relative degree greater than -1; H2 undefined")
        A, B, C = tf2ss(self.num, self.den)
        return h2_norm(A, B, C, sol = sol)

    def get_re_im(self, param_sym):
        den = deepcopy(self.den)
        den_cc = deepcopy(self.den)
        num = deepcopy(self.num)
        # Make the complex conjugate of denominator
        for i in range(1, len(den)):
            if i % 2 == 1: # odd power of 's' 
                den_cc[-i-1] = -den_cc[-i-1]

        # multiply num and den by complex conjugate
        num_cc = con(self.num, den_cc)
        den_cc = con(self.den, den_cc)

        # Substitute s = j\omega (turn those j^2, j^4,... into -1^n/2)
        for i in range(0, len(num_cc)): # start at 1 to skip the 0th power of j\omega
            number_of_i_squared = ca.floor(i/2)
            if number_of_i_squared % 2 == 1:   # odd number of i^2 -> -1
                num_cc[-i-1] *= -1.0
        for i in range(0, len(den_cc)):
            num_i_squared = ca.floor(i/2)
            if num_i_squared % 2 == 1: # odd number of i^2 -> -1
                den_cc[-i-1] *= -1.0

        imag_coeff = deepcopy(num_cc)
        real_coeff = deepcopy(num_cc)
        for i in range(len(num_cc)):
            if i % 2 == 0: # even power of 's', 
                imag_coeff[-i-1] = ca.MX(0)
            else:
                real_coeff[-i-1] = ca.MX(0)
        omega = ca.MX.sym('omega')
        imag_poly = ca.polyval(ca.vertcat(*imag_coeff),omega)
        real_poly = ca.polyval(ca.vertcat(*real_coeff),omega)
        den_poly = ca.polyval(ca.vertcat(*den_cc),omega)
        imag_fn = ca.Function('imag_fn', [omega, *param_sym], [imag_poly/den_poly])
        real_fn = ca.Function('real_fn', [omega, *param_sym], [real_poly/den_poly])
        return real_fn, imag_fn

    def nyquist(self, param_sym, param_num):
        real_fn, imag_fn = self.get_re_im(param_sym)
        plt.figure()
        plt.clf()
        for om in np.logspace(-3,5, num = 1000):
            plt.plot(real_fn(om, *param_num), imag_fn(om, *param_num),'ko')
        
        '''
        r1, r2 = self.critical_points(imag_coeff, den_cc)
        r1_fn = ca.Function('r1', [*param_sym], [r1])
        r2_fn = ca.Function('r2', [*param_sym], [r2])
        r1_num = r1_fn(*param_num)
        r2_num = r2_fn(*param_num)
        print('roots: {} {}'.format(r1_num, r2_num))
        plt.plot(real_fn(r1_num, *param_num), imag_fn(6.28*r1_num, *param_num),'rx')
        plt.plot(real_fn(r2_num, *param_num), imag_fn(6.28*r2_num, *param_num),'rx')
        '''

    def critical_points(self, num, den):
        # Find points where the slope of the given polynomial are zero
        d_num = der(num)
        d_den = der(den)
        d_num_total = lsub(con(den, d_num), con(num, d_den))
        print(con(den, d_num))
        print(con(d_den, num))
        poly_coeffs = ca.MX.sym('poly',3,1)
        #print(d_coeffs)
        #a = d_coeffs[0]
        #b = d_coeffs[2]
        #c = d_coeffs[4]
        #print('a {} b {} c{}'.format(a, b, c))
        #r1 = ca.sqrt(-(-b+ca.sqrt(b**2-4*a*c))/(2*a))
        #r2 = ca.sqrt(-(-b-ca.sqrt(b**2-4*a*c))/(2*a))
        return r1, r2


class lyap_solver(ca.Callback):
    from scipy.linalg import solve_continuous_lyapunov
    # Following the /test/python/function.py
    def __init__(self, name, d, opts = {}):
        ca.Callback.__init__(self)
        self.d = d
        self.construct(name, opts)

    def get_n_in(self):  return 2
    def get_n_out(self): return 1
    def get_sparsity_in(self, i): return ca.Sparsity.dense(self.d, self.d)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(self.d, self.d)

    def has_forward(self, nfwd): return True
    def has_reverse(self, nadj): return True

    def init(self):
        print('initializing')

    def eval(self, arg):
        from scipy.linalg import solve_continuous_lyapunov
        A = arg[0]
        Q = arg[1]
        return [solve_continuous_lyapunov(A, -Q)]

    def get_forward(self, nfwd, name, inames, onames, opts):
        assert(nfwd==1)
        class ForwardFun(ca.Callback):
            
            def __init__(self, d):
                ca.Callback.__init__(self)
                self.d = d
                self.construct(name, opts)
            def get_n_in(self):  return 5
            def get_n_out(self): return 1
            def get_sparsity_in(self, i): return ca.Sparsity.dense(self.d, self.d)
            def get_sparsity_out(self, i): return ca.Sparsity.dense(self.d, self.d)

            def eval(self, arg):
                # The argument order seems to be: args to f(), solution to f(), then the deriv of inputs to f().
                from scipy.linalg import solve_continuous_lyapunov # Sorry, but the C++ bindings cant find if we import above
                A = arg[0]
                Q = arg[1]
                P = arg[2]
                Ad = arg[3]
                Qd = arg[4]
                Pd = solve_continuous_lyapunov(-A, (Ad @ P + P @ Ad + Qd))
                return [Pd]

        self.cb_fwd = ForwardFun(self.d)
        return self.cb_fwd

    def get_reverse(self, nadj, name, inames, onames, opts):
        assert(nadj==1)
        class BackwardFun(ca.Callback):
            def __init__(self, d):
                ca.Callback.__init__(self)
                self.d = d
                self.construct(name, opts)
            def get_n_in(self):  return 4
            def get_n_out(self): return 2
            def get_sparsity_in(self, i): return ca.Sparsity.dense(self.d, self.d)
            def get_sparsity_out(self, i): return ca.Sparsity.dense(self.d, self.d)
            def eval(self, arg):
                # The argument order seems to be: args to f(), solution to f(), then the deriv of inputs to f().
                from scipy.linalg import solve_continuous_lyapunov # Sorry, but the C++ bindings cant find if we import above
                A = arg[0]
                Q = arg[1]
                P = arg[2]
                Pbar = arg[3]
                S = solve_continuous_lyapunov(A, -Pbar)
                return [S @ P.T + S.T @ P, S]

        self.cb_rev = BackwardFun(self.d)
        return self.cb_rev


def con(a_in, b_in):
# Convolve the lists a_in and b_in; assumes + and * defined over the elements
    a = deepcopy(a_in)
    b = deepcopy(b_in)
    ia = len(a)
    ib = len(b)
    nl = ia+ib-1
    for j in range(ia,nl):
        a.append(0.0)
    for j in range(ib,nl):
        b.append(0.0)
    ret = []
    for i in range(nl):
        new_el = 0
        for j in range(nl):
            new_el += a[j]*b[i-j]
        ret.append(new_el)
    return ret

def ladd(a_in, b_in):
    a = deepcopy(a_in)
    b = deepcopy(b_in)
    n = max(len(a),len(b))
    a = [0]*(n-len(a)) + a
    b = [0]*(n-len(b)) + b
    return [aa+bb for aa, bb in zip(a,b)]


def lsub(a_in, b_in):
    a = deepcopy(a_in)
    b = deepcopy(b_in)
    n = max(len(a),len(b))
    a = [0]*(n-len(a)) + a
    b = [0]*(n-len(b)) + b
    return [aa-bb for aa, bb in zip(a,b)]



def fb(G, C):
# Return G/(1+GC), negative FB
    new_num = con(G.num, C.den)
    new_den = ladd(con(G.den, C.den), con(G.num, C.num))
    return sys(new_num, new_den)

def der(a):
# If a is coeffs of a polynomial, returns the derivative
    l = len(a)
    ret = []
    for i in range(l-1):
        ret.append((l-i-1)*a[i])
    return ret

def instab(a): 
# Returns all nonzero entries in the first column of Routh-Hurwitz table
# This can then be used as a stability condition - that the sign all match
    rt = [] # Routh table
    rt.append([*a[0::2], 0.0, 0.0, 0.0])
    rt.append([*a[1::2], 0.0, 0.0, 0.0])
    for i in range(2,len(a)):
        new_row = []
        for j in range(len(rt[i-2])-2):
            new_row.append((rt[i-1][0]*rt[i-2][j+1]-rt[i-2][0]*rt[i-1][j+1])/rt[i-1][0])
        new_row.append(0.0)
        if new_row[0].is_zero:
            break
        rt.append(new_row)
    rt_first_col_sign = []
    for row in rt:
        rt_first_col_sign.append(ca.sign(row[0]))
    #return len(rt_first_col_sign) - ca.sum1(rt_first_col_sign) # this just returns the # of sign changes
    return [row[0] for row in rt]

def tf2ss(num_in, den_in):
# Put system into controllable canonical form, where num and den are lists of of coeffs on s
    num = deepcopy(num_in)
    den = deepcopy(den_in)

    # Check if we can cancel some poles/zeros at 0
    while ca.MX.is_zero(num[-1]) and ca.MX.is_zero(den[-1]):
        print('Cancelling pole/zero at 0')
        num.pop()
        den.pop() 

    # Pad front of numerator so they're the same length
    n = len(den)-1
    num = [0]*(n-len(num)) + num

    # Scale so leading coeff on den is 1
    num = [nu/den[0] for nu in num]
    den = [de/den[0] for de in den]

    Am = ca.vertcat(ca.horzcat(ca.MX.zeros((n-1,1)), ca.MX.eye(n-1)), ca.MX.zeros((1,n)))
    Bm = ca.MX.zeros((n,1))
    Cm = ca.MX.zeros((1,n))
    for i in range(n):
        Am[-1, i] = -den[-i-1]
        Cm[0,i] = num[-i-1]
    Bm[-1] = 1
    return Am, Bm, Cm

def lyap(A, Q, sol='symbolicqr'):
# Solve the Lyapunov equation A'X+XA = Q for real A in R n x n
    n = A.shape[0]
    if not A.shape[1] == n or not Q.shape[0] == n or not Q.shape[1] == n:
        print('ERROR in lyap: A and Q must be square of same dim')
    vec_Q = ca.reshape(Q, n*n, 1)
    vec_IA_AI = ca.kron(np.eye(n),A) + ca.kron(A, np.eye(n))
    linsolver = ca.Linsol('lyapsol',sol, vec_IA_AI.sparsity(), {})
    vec_X = linsolver.solve(vec_IA_AI, -vec_Q) 
    return ca.reshape(vec_X, n, n)

def rev_mode_lyap(A, Q, P, P_bar):
# Following "Automatic differentiation of Sylvester, Lyapunov and algebraic Riccati equations" from Kao and Hennequin
# Not actually necessary (can just do autodiff through lyap)
    S = lyap(A.T, P_bar)
    A_bar = S @ P.T + S.T @ P
    Q_bar = S
    return A_bar, Q_bar

def h2_norm(A, B, C, sol = 'lapacklu'):
# Calculate the H2 norm of the system by the observability grammian 
    Xo = lyap(A.T, C.T @ C, sol = sol)
    return ca.sqrt(ca.trace(B.T @ Xo @ B))

def rev_mode_riccati(A, B, C, Q, P, P_bar):
    A_tilde = A-B@K


def pade(dt, N):
# Returns num/den for a 3rd order pade approx
# See, e.g. "Rational Approximation of Time Delay" by Hanta
    if N == 1:
        num = [-dt, 2]
        den = [dt, 2]
    elif N == 2:
        num = [dt**2, -4*dt, 8]
        den = [dt**2, 4*dt, 8]
    elif N == 3:
        num = [-dt**3, 6*dt**2, -24*dt,  48]
        den = [dt**3, 6*dt**2, 24*dt, 48]
    else:
        num = [dt**4, -8*dt**3, 48*dt**2, -192*dt, 384]
        den = [dt**4, 8*dt**3, 48*dt**2, 192*dt, 384]
    if N > 4:
        print('N > 4 not supported, just giving N = 4')
    return sys(num, den)
