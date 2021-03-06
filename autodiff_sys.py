# Description:   Class for dynamic systems in Casadi, supports symbolic/automatic differentiation and optimization
# Contributor:   Kevin Haninger {khaninger@gmail.com}
# First written: 2021.07.07


# Library imports
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Project imports
from .helper_fns import *

class Sys():
    def __init__(self, num, den, symb_type = ca.MX):
        '''
        Initalize a transfer function system with coefficients given in num(erator) and den(ominator).
        The lists are ordered from highest power of 's' to lowest
        '''
        self.ty = symb_type
        self.num = [symb_type(n) for n in num]
        self.den = [symb_type(d) for d in den]

    def __mul__(self, other): # Overload * operator
        new_num = con(self.num, other.num)
        new_den = con(self.den, other.den)
        return Sys(new_num, new_den)

    def __rmul__(self, other): # Overload right * operator
        new_num = con(self.num, other.num)
        new_den = con(self.den, other.den)
        return Sys(new_num, new_den)

    def __truediv__(self, other): # Overload the / operator
        new_num = con(self.num, other.den)
        new_den = con(self.den, other.num)
        return Sys(new_num, new_den)

    def __add__(self, other): # Overload the + operator
        if isinstance(other, Sys):
            new_num = ladd(con(self.num, other.den), con(self.den, other.num))
            new_den = con(self.den, other.den)
        elif other == 1:
            new_num = ladd(self.den, self.num)
            new_den = self.den
        return Sys(new_num, new_den)

    def __radd__(self, other): # Overload the right + operator
        if isinstance(other, Sys):
            new_num = ladd(con(self.num, other.den), con(self.den, other.num))
            new_den = con(self.den, other.den)
        elif other == 1:
            new_num = ladd(self.den, self.num)
            new_den = self.den
        return Sys(new_num, new_den)

    def __str__(self):
        return 'num: ' + str(self.num) + ' den: ' + str(self.den)

    def h2(self, sol = 'scipy'):
        '''
        Calculates and returns the H2 norm for the current System
        Parameters:
         - sol: if 'scipy', the Sylvester solver from scipy is used,
                   which uses the Bartel-Stewart algorithm from LAPACK
                   This is usually more robust for poorly conditioned Systems
                if 'lapackqr', or 'ldl' or other argument from ca.linsol solver
                   Lyap is solved with the naive vectorize -> linear solve
        Returns:
         - h2_result: the H2 norm of the System
        Note:
         - Poles/zeros at 0 are cancelled automatically
         - Checks relative degree to see if norm exists
        '''
        while self.ty.is_zero(self.num[-1]) and self.ty.is_zero(self.den[-1]):
            print('Cancelling pole/zero at 0')
            self.num.pop()
            self.den.pop() 
        if self.ty.is_zero(self.den[-1]):
            print("System has pole at zero; H2 undefined")
        if len(self.num) >= len(self.den):
            print("System has relative degree greater than -1; H2 undefined")
        A, B, C = tf2ss(self.num, self.den)
        h2_result, self.solver = h2(A, B, C, sol = sol)
        return h2_result

    def get_re_im(self, param_sym):
        '''
        Returns the real and imaginary components of the System as Casadi functions
        These functions should be called with arguments (omega, *param_num), where
        param_num is a list of parameter values aligned with param_sym.
        Parameters:
         - param_sym: a list with the symbolic variables which are in the System
        Returns:
         - real_fn: a Casadi function for the real component of the Sys over freq (rad/sec)
         - imag_fn: "                       " imag "                                      "
        '''
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
            if i % 2 == 0: # even power of 's' -> it's real, eliminate from imag
                imag_coeff[-i-1] = ca.MX(0) 
            else: # odd power of 's' -> it's imag, eliminate from real
                real_coeff[-i-1] = ca.MX(0)
        omega = ca.MX.sym('omega')
        imag_poly = ca.polyval(ca.vertcat(*imag_coeff),omega)
        real_poly = ca.polyval(ca.vertcat(*real_coeff),omega)
        den_poly = ca.polyval(ca.vertcat(*den_cc),omega)
        imag_fn = ca.Function('imag_fn', [omega, *param_sym], [imag_poly/den_poly])
        real_fn = ca.Function('real_fn', [omega, *param_sym], [real_poly/den_poly])
        return real_fn, imag_fn

    def nyquist(self, param_sym, param_num, om_range = [-3, 5]):
        '''
        Nyquist plot
        Parameters:
         - param_sym: a list of all symbolic variables in system
         - param_num: the numerical values param_sym should be eval at
         - om_range: freq range in log10 rad/sec
        '''
        real_fn, imag_fn = self.get_re_im(param_sym)
        plt.figure()
        plt.clf()
        for om in np.logspace(*om_range, num = 1000):
            plt.plot(real_fn(om, *param_num), imag_fn(om, *param_num),'ko')

    def critical_points(self, num, den):
        # Find points where the slope of the given polynomial are zero
        # Not currently working!  Also needs to generalize
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
        return r1, r2
