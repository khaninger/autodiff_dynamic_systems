import casadi as ca

s = ca.SX.sym('s')

M = ca.SX.sym('M')
B = ca.SX.sym('B')
Kp = ca.SX.sym('Kp')
Kd = ca.SX.sym('Kd')

R = (Kd*s+Kp)/(Ms**2+B*s+Kp)

D = 1.0

Ka = ca.SX.sym('Ka')
Cff = 1+Ka*s

Ma = ca.SX.sym('Ma')
Ba = ca.SX.sym('Ba')
A = 1/(Ma*s**2+ Ba*s)

