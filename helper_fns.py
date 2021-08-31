# Description:   Helper functions for manipulating transfer functions, etc
# Contributor:   Kevin Haninger {khaninger@gmail.com}
# First written: 2021.07.07

# System imports
from copy import deepcopy

# Library imports
import casadi as ca
import numpy as np

# Project imports
from lyap_solver import LyapSolver

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
# add two lists such that the last element is aligned
    a = deepcopy(a_in)
    b = deepcopy(b_in)
    n = max(len(a),len(b))
    a = [0]*(n-len(a)) + a
    b = [0]*(n-len(b)) + b
    return [aa+bb for aa, bb in zip(a,b)]


def lsub(a_in, b_in):
# subtract two lists such that the last element is aligned
    a = deepcopy(a_in)
    b = deepcopy(b_in)
    n = max(len(a),len(b))
    a = [0]*(n-len(a)) + a
    b = [0]*(n-len(b)) + b
    return [aa-bb for aa, bb in zip(a,b)]

def der(a):
# If a is coeffs of a polynomial, returns the derivative
    l = len(a)
    ret = []
    for i in range(l-1):
        ret.append((l-i-1)*a[i])
    return ret

def routh_table(a): 
# Returns all nonzero entries in the first column of Routh-Hurwitz table
# Doesn't handle all cases of 0s!
# This can then be used as a stability condition - that the sign all match
    rt = [] # Routh table
    rt.append([*a[0::2], 0.0, 0.0, 0.0])
    rt.append([*a[1::2], 0.0, 0.0, 0.0])
    for i in range(2,len(a)):
        new_row = []
        for j in range(len(rt[i-2])-2):
            new_row.append((rt[i-1][0]*rt[i-2][j+1]-rt[i-2][0]*rt[i-1][j+1])/rt[i-1][0])
        if type(new_row[0]) is ca.MX:
            if new_row[0].is_zero():
                print('Routh Table has leading sym 0; interrupting, may not be complete')
                break
        elif new_row[0] == 0:
            print('Routh Table has leading 0; interrupting, may not be complete')
            break
        new_row.append(0.0)
        rt.append(new_row)

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

def pade(dt, N):
# Returns num/den for a 3rd order pade approx
# See, e.g. "Rational Approximation of Time Delay" by Hanta
    if N == 1:
        num = [-dt, 2]
        den = [dt, 2]
    elif N == 2:
        num = [dt**2, -6*dt, 12]
        den = [dt**2,  6*dt, 12]
    elif N == 3:
        num = [-dt**3, 12*dt**2, -60*dt,  120]
        den = [ dt**3, 12*dt**2,  60*dt, 120]
    else:
        num = [dt**4, -20*dt**3, 180*dt**2, -840*dt, 1680]
        den = [dt**4,  20*dt**3, 180*dt**2,  840*dt, 1680]
    if N > 4:
        print('N > 4 not supported, just giving N = 4')
    return num, den

def lyap(A, Q, sol='scipy'):
    # Solve the Lyapunov equation A'X+XA = Q for real A in R n x n
    n = A.shape[0]
    if not A.shape[1] == n or not Q.shape[0] == n or not Q.shape[1] == n:
        print('ERROR in lyap: A and Q must be square of same dim')
    if sol is 'scipy':
        solver = LyapSolver('lyap_solver', A.shape[0])
        X = solver(A, Q)
    else:
    # Solve with the casadi solve. Not as num stable for poorly-conditioned A/Q.
        vec_Q = ca.reshape(Q, n*n, 1)
        vec_IA_AI = ca.kron(np.eye(n),A) + ca.kron(A, np.eye(n))
        solver = ca.Linsol('lyapsol',sol, vec_IA_AI.sparsity(), {})
        vec_X = solver.solve(vec_IA_AI, -vec_Q)
        X = ca.reshape(vec_X, n, n)
    return X, solver # solver needs to be kept for scipy to call reverse/forward later.

def h2(A, B, C, sol = 'scipy'):
    Xo, solver = lyap(A.T, C.T @ C, sol = sol)
    h2 = ca.sqrt(ca.trace(B.T @ Xo @ B))
    return h2, solver

import csv
def import_csv(name):
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        f = []
        x = []
        for row in csv_reader:
            x.append(float(row[0])/1000.0)
            f.append(float(row[1]))
    return x,f
