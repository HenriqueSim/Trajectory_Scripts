import casadi as ca
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ConstraintType:
    Position = "pos"
    Velocity = "vel"
    Acceleration = "acc"
    Jerk = "jerk"

class Constraint:
    def __init__(self, time, type_, value):
        self.time = time
        self.type = type_
        self.value = value

def generate_polynomial_trajectory_optimized(constraints, order):
    n_coeffs = order + 1
    coeffs = ca.SX.sym('c', n_coeffs)
    t0 = constraints[0].time
    tf = constraints[-1].time
    t = ca.SX.sym('t')

    snap = 0
    for i in range(4, n_coeffs):
        snap += i * (i-1) * (i-2) * (i-3) * coeffs[i] * t**(i-4)
    cost = 0
    N_int = 50
    dt = (tf - t0) / N_int
    for i in range(N_int + 1):
        ti = t0 + i * dt
        cost += ca.substitute(snap**2, t, ti) * dt

    g = []
    g_val = []
    for c in constraints:
        t_vec = np.array([c.time**i for i in range(n_coeffs)])
        if c.type == ConstraintType.Position:
            g.append(ca.dot(coeffs, t_vec))
        elif c.type == ConstraintType.Velocity:
            v_vec = np.array([0 if i == 0 else i * c.time**(i-1) for i in range(n_coeffs)])
            g.append(ca.dot(coeffs, v_vec))
        elif c.type == ConstraintType.Acceleration:
            a_vec = np.array([0 if i <= 1 else i*(i-1)*c.time**(i-2) for i in range(n_coeffs)])
            g.append(ca.dot(coeffs, a_vec))
        elif c.type == ConstraintType.Jerk:
            j_vec = np.array([0 if i <= 2 else i*(i-1)*(i-2)*c.time**(i-3) for i in range(n_coeffs)])
            g.append(ca.dot(coeffs, j_vec))
        g_val.append(c.value)

    nlp = {'x': coeffs, 'f': cost, 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(lbg=g_val, ubg=g_val, lbx=-ca.inf, ubx=ca.inf, x0=np.zeros(n_coeffs))
    return np.array(sol['x']).flatten()

def create_dynamic_constraints(start_point, end_point, window_center, window_direction, margin=0.2, T1=2.0, T2=0.4, T3=2.0):
    T_final = T1 + T2 + T3
    t0, t1, t2, t3 = 0.0, T1, T1 + T2, T1 + T2 + T3
    t_middle = (t1 + t2) / 2

    window_dir = np.array(window_direction)
    window_dir /= np.linalg.norm(window_dir)

    constraints = {'x': [], 'y': [], 'z': []}
    for i, axis in enumerate(['x', 'y', 'z']):
        acc_window = window_dir[i]

        # Segment 1
        constraints[axis].append([
            Constraint(t0, ConstraintType.Position, start_point[i]),
            Constraint(t0, ConstraintType.Velocity, 0.0),
            Constraint(t0, ConstraintType.Acceleration, 0.0),
            # Constraint(t1, ConstraintType.Acceleration, 0.0),
            Constraint(t1, ConstraintType.Jerk, 0.0),
        ])

        # Segment 2 (window)
        constraints[axis].append([
            Constraint(t1, ConstraintType.Acceleration, 0.0),
            Constraint(t_middle, ConstraintType.Acceleration, acc_window),
            Constraint(t2, ConstraintType.Acceleration, 0.0),
        ])

        # Segment 3
        constraints[axis].append([
            Constraint(t2, ConstraintType.Acceleration, 0.0),
            Constraint(t2, ConstraintType.Jerk, 0.0),
            Constraint(t3, ConstraintType.Position, end_point[i]),
            Constraint(t3, ConstraintType.Velocity, 0.0),
            Constraint(t3, ConstraintType.Acceleration, 0.0)
        ])
    return constraints, T1, T2, T3

def evaluate_polynomial(coeffs, t):
    return sum(coeffs[i] * t**i for i in range(len(coeffs)))

def evaluate_velocity(coeffs, t):
    return sum(i * coeffs[i] * t**(i - 1) for i in range(1, len(coeffs)))

def evaluate_acceleration(coeffs, t):
    return sum(i * (i - 1) * coeffs[i] * t**(i - 2) for i in range(2, len(coeffs)))

def eval_series(coeffs, ts, eval_fn):
    return [eval_fn(coeffs, t) for t in ts]

def eval_derivatives(coeffs, t, order):
    v = sum(i * coeffs[i] * t**(i - 1) for i in range(1, order+1))
    a = sum(i * (i - 1) * coeffs[i] * t**(i - 2) for i in range(2, order+1))
    j = sum(i * (i - 1) * (i - 2) * coeffs[i] * t**(i - 3) for i in range(3, order+1))
    return v, a, j

# User-defined input
start_point = [-2.0, 0.0, -1.5]
end_point = [2.0, 0.0, -1.5]
window_center = [0.0, 0.0, -2.0]
window_direction = [1.0, 0.0, 0.0]  # Downward-right inclination

# Generate dynamic constraints
constraints_dict, T1, T2, T3 = create_dynamic_constraints(start_point, end_point, window_center, window_direction)
T_final = T1 + T2 + T3

# Generate trajectories
orders = [9, 5, 9]
axes_coeffs = {'x': [], 'y': [], 'z': []}
for axis in ['x', 'y', 'z']:
    for i in range(3):
        coeffs = generate_polynomial_trajectory_optimized(constraints_dict[axis][i], orders[i])
        axes_coeffs[axis].append(coeffs)

# Evaluate
t1 = np.linspace(0.0, T1, 100)
t2 = np.linspace(T1, T1 + T2, 100)
t3 = np.linspace(T1 + T2, T_final, 100)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([evaluate_polynomial(axes_coeffs['x'][0], t) for t in t1],
        [evaluate_polynomial(axes_coeffs['y'][0], t) for t in t1],
        [evaluate_polynomial(axes_coeffs['z'][0], t) for t in t1], label='Segment 1', color='blue')
ax.plot([evaluate_polynomial(axes_coeffs['x'][1], t) for t in t2],
        [evaluate_polynomial(axes_coeffs['y'][1], t) for t in t2],
        [evaluate_polynomial(axes_coeffs['z'][1], t) for t in t2], label='Segment 2', color='orange')
ax.plot([evaluate_polynomial(axes_coeffs['x'][2], t) for t in t3],
        [evaluate_polynomial(axes_coeffs['y'][2], t) for t in t3],
        [evaluate_polynomial(axes_coeffs['z'][2], t) for t in t3], label='Segment 3', color='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory')
ax.legend()
plt.tight_layout()
plt.show()
