import casadi as ca
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ConstraintType:
    Position = "pos"
    Velocity = "vel"
    Acceleration = "acc"
    Jerk = "jerk"  # New type for jerk constraints

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

def eval_derivatives(coeffs, t, order):
    v = sum(i * coeffs[i] * t**(i - 1) for i in range(1, order+1))
    a = sum(i * (i - 1) * coeffs[i] * t**(i - 2) for i in range(2, order+1))
    j = sum(i * (i - 1) * (i - 2) * coeffs[i] * t**(i - 3) for i in range(3, order+1))
    return v, a, j

def evaluate_polynomial(coeffs, t):
    return sum(coeffs[i] * t**i for i in range(len(coeffs)))

def evaluate_velocity(coeffs, t):
    return sum(i * coeffs[i] * t**(i - 1) for i in range(1, len(coeffs)))

def evaluate_acceleration(coeffs, t):
    return sum(i * (i - 1) * coeffs[i] * t**(i - 2) for i in range(2, len(coeffs)))

def evaluate_jerk(coeffs, t):
    return sum(i * (i - 1) * (i - 2) * coeffs[i] * t**(i - 3) for i in range(3, len(coeffs)))

# Define trajectory order and time segments
order = 9
T1 = 0.8
T2 = 0.2015
# T2 = 0.2
T3 = 0.8
T_final = T1 + T2 + T3

# X-axis constraints with jerk continuity
c1_x = generate_polynomial_trajectory_optimized([
    Constraint(0.0, ConstraintType.Position, -2.0),
    Constraint(0.0, ConstraintType.Velocity, 0.0),
    Constraint(0.0, ConstraintType.Acceleration, 0.0),
    Constraint(T1, ConstraintType.Position, -0.3),
    Constraint(T1, ConstraintType.Acceleration, 0.0),
    Constraint(T1, ConstraintType.Jerk, 0.0)
], order)
v1_x, a1_x, j1_x = eval_derivatives(c1_x, T1, order)

c2_x = generate_polynomial_trajectory_optimized([
    Constraint(T1, ConstraintType.Position, -0.3),
    Constraint(T1, ConstraintType.Velocity, v1_x),
    Constraint(T1, ConstraintType.Acceleration, a1_x),
    Constraint(T1 + T2/2, ConstraintType.Acceleration, 0.0),
    Constraint(T1 + T2, ConstraintType.Position, 0.3),
], 5)
v2_x, a2_x, j2_x = eval_derivatives(c2_x, T1 + T2, 5)

c3_x = generate_polynomial_trajectory_optimized([
    Constraint(T1 + T2, ConstraintType.Position, 0.3),
    Constraint(T1 + T2, ConstraintType.Velocity, v2_x),
    Constraint(T1 + T2, ConstraintType.Acceleration, a2_x),
    Constraint(T1 + T2, ConstraintType.Jerk, j2_x),
    Constraint(T_final, ConstraintType.Position, 2.0),
    Constraint(T_final, ConstraintType.Velocity, 0.0),
    Constraint(T_final, ConstraintType.Acceleration, 0.0)
], order)

# Y-axis constraints with jerk continuity
c1_y = generate_polynomial_trajectory_optimized([
    Constraint(0.0, ConstraintType.Position, 0.0),
    Constraint(0.0, ConstraintType.Velocity, 0.0),
    Constraint(0.0, ConstraintType.Acceleration, 0.0),
    Constraint(T1, ConstraintType.Position, 0.0),
    Constraint(T1, ConstraintType.Acceleration, 0.0),
    Constraint(T1, ConstraintType.Jerk, 0.0)
], order)
v1_y, a1_y, j1_y = eval_derivatives(c1_y, T1, order)

c2_y = generate_polynomial_trajectory_optimized([
    Constraint(T1, ConstraintType.Position, 0.0),
    Constraint(T1, ConstraintType.Velocity, v1_y),
    Constraint(T1, ConstraintType.Acceleration, a1_y),
    Constraint(T1 + T2/2, ConstraintType.Acceleration, 0.0),
    Constraint(T1 + T2, ConstraintType.Position, 0.0)
], order)
v2_y, a2_y, j2_y = eval_derivatives(c2_y, T1 + T2, 5)

c3_y = generate_polynomial_trajectory_optimized([
    Constraint(T1 + T2, ConstraintType.Position, 0.0),
    Constraint(T1 + T2, ConstraintType.Velocity, v2_y),
    Constraint(T1 + T2, ConstraintType.Acceleration, a2_y),
    Constraint(T1 + T2, ConstraintType.Jerk, j2_y),
    Constraint(T_final, ConstraintType.Position, 0.0),
    Constraint(T_final, ConstraintType.Velocity, 0.0),
    Constraint(T_final, ConstraintType.Acceleration, 0.0)
], order)

# Z-axis constraints with jerk continuity
c1_z = generate_polynomial_trajectory_optimized([
    Constraint(0.0, ConstraintType.Position, -1.5),
    Constraint(0.0, ConstraintType.Velocity, 0.0),
    Constraint(0.0, ConstraintType.Acceleration, 0.0),
    Constraint(T1, ConstraintType.Position, -2.0),
    Constraint(T1, ConstraintType.Velocity, 0.0),
    Constraint(T1, ConstraintType.Acceleration, 0.0)
], order)
v1_z, a1_z, j1_z = eval_derivatives(c1_z, T1, order)

c2_z = generate_polynomial_trajectory_optimized([
    Constraint(T1, ConstraintType.Position, -2.0),
    # Constraint(T1, ConstraintType.Velocity, v1_z),
    Constraint(T1 + T2, ConstraintType.Position, -2.0)
], 1)       # order 1 to produce a constant velocity 
v2_z, a2_z, j2_z = eval_derivatives(c2_z, T1 + T2, 1)

c3_z = generate_polynomial_trajectory_optimized([
    Constraint(T1 + T2, ConstraintType.Position, -2.0),
    Constraint(T1 + T2, ConstraintType.Velocity, v2_z),
    Constraint(T1 + T2, ConstraintType.Acceleration, a2_z),
    Constraint(T1 + T2, ConstraintType.Jerk, j2_z),
    Constraint(T_final, ConstraintType.Position, -1.5),
    Constraint(T_final, ConstraintType.Velocity, 0.0),
    Constraint(T_final, ConstraintType.Acceleration, 0.0)
], order)


# Save coefficients to YAML file
trajectory_data = {
    'x_axis': {
        'segment_1': c1_x.tolist(),
        'segment_2': c2_x.tolist(),
        'segment_3': c3_x.tolist()
    },
    'y_axis': {
        'segment_1': c1_y.tolist(),
        'segment_2': c2_y.tolist(),
        'segment_3': c3_y.tolist()
    },
    'z_axis': {
        'segment_1': c1_z.tolist(),
        'segment_2': c2_z.tolist(),
        'segment_3': c3_z.tolist()
    },
    'timing': {
        'T1': T1,
        'T2': T2,
        'T3': T3
    }
}

with open("polynomial_trajectory.yaml", "w") as file:
    yaml.dump(trajectory_data, file, default_flow_style=False)
print("Trajectory coefficients saved to 'polynomial_trajectory.yaml'")



# Time samples
t1 = np.linspace(0.0, T1, 100)
t2 = np.linspace(T1, T1 + T2, 100)
t3 = np.linspace(T1 + T2, T_final, 100)

# 3D Trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([evaluate_polynomial(c1_x, t) for t in t1],
        [evaluate_polynomial(c1_y, t) for t in t1],
        [evaluate_polynomial(c1_z, t) for t in t1], label='Segment 1', color='blue')
ax.plot([evaluate_polynomial(c2_x, t) for t in t2],
        [evaluate_polynomial(c2_y, t) for t in t2],
        [evaluate_polynomial(c2_z, t) for t in t2], label='Segment 2', color='orange')
ax.plot([evaluate_polynomial(c3_x, t) for t in t3],
        [evaluate_polynomial(c3_y, t) for t in t3],
        [evaluate_polynomial(c3_z, t) for t in t3], label='Segment 3', color='green')

ax.set_xlabel('X')
ax.set_ylabel('Y (fixed at 0)')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory with Segment Coloring')
ax.legend()
plt.tight_layout()

# Velocity and acceleration
def eval_series(coeffs, ts, eval_fn):
    return [eval_fn(coeffs, t) for t in ts]

vel_x = np.concatenate([
    eval_series(c1_x, t1, evaluate_velocity),
    eval_series(c2_x, t2, evaluate_velocity),
    eval_series(c3_x, t3, evaluate_velocity)
])
vel_y = np.concatenate([
    eval_series(c1_y, t1, evaluate_velocity),
    eval_series(c2_y, t2, evaluate_velocity),
    eval_series(c3_y, t3, evaluate_velocity)
])
vel_z = np.concatenate([
    eval_series(c1_z, t1, evaluate_velocity),
    eval_series(c2_z, t2, evaluate_velocity),
    eval_series(c3_z, t3, evaluate_velocity)
])
acc_x = np.concatenate([
    eval_series(c1_x, t1, evaluate_acceleration),
    eval_series(c2_x, t2, evaluate_acceleration),
    eval_series(c3_x, t3, evaluate_acceleration)
])
acc_y = np.concatenate([
    eval_series(c1_y, t1, evaluate_acceleration),
    eval_series(c2_y, t2, evaluate_acceleration),
    eval_series(c3_y, t3, evaluate_acceleration)
])
acc_z = np.concatenate([
    eval_series(c1_z, t1, evaluate_acceleration),
    eval_series(c2_z, t2, evaluate_acceleration),
    eval_series(c3_z, t3, evaluate_acceleration)
])
time_vals = np.concatenate([t1, t2, t3])

# Plot velocities and accelerations
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_vals, vel_x, label='x velocity', color='blue')
plt.plot(time_vals, vel_y, label='y velocity', color='orange')
plt.plot(time_vals, vel_z, label='z velocity', color='green')
plt.title('Velocity')
plt.ylabel('Velocity')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_vals, acc_x, label='x acceleration', color='blue')
plt.plot(time_vals, acc_y, label='y acceleration', color='orange')
plt.plot(time_vals, acc_z, label='z acceleration', color='green')
plt.title('Acceleration')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
