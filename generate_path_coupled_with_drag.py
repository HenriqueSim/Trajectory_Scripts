#!/usr/bin/env python3
"""
Single-segment polynomial trajectory with:
 - midpoint constraints: position, roll (phi) and Euler-rates (phi_dot, theta_dot)
 - exact nonlinear coupling (NED)
 - derivative-level (jerk -> Euler-rates) exact linear relation enforced at midpoint
 - minimize integrated snap^2
 - enforce a_z(t) <= g - margin at M_samples times (no 'negative thrust' requirement)
Produces plots for velocities, accelerations, orientations, u and angular rates.

Updated: linear drag per-axis f_drag = -rho * v (in inertial frame)
"""
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import yaml
from mpl_toolkits.mplot3d import Axes3D

# ---------------- User parameters ----------------
order = 11
n = order + 1
t0 = 0.0
tf = 2.75
t_mid = 1.375
grav = 9.81

phi_deg = -38.5
phi_des = np.deg2rad(phi_deg)   # desired roll at midpoint
theta_deg_des = 0.0
theta_des = np.deg2rad(theta_deg_des)  # desired pitch at midpoint 
psi_des = 0.0                   # yaw fixed
theta0 = 0.0                    # initial pitch guess

# Desired Euler-rate at midpoint (rad/s).
phi_dot_des = 0.0    # desired roll rate at t_mid
phi_dot_0 = 0.0  # initial roll rate guess
theta_dot_des = 0.0  # desired pitch rate at t_mid
theta_dot_0 = 0.0  # initial pitch rate guess

# Start/end positions
x_start, x_end = 2.0, -2.0
y_start, y_end = 0.0, 0.0
z_start, z_end = -1.5, -1.5

# Start/end velocities, accelerations and jerks
v0, vf = 0.0, 0.0
a0, af = 0.0, 0.0

# Midpoint (position)
x_mid, y_mid, z_mid = 0.188, -0.507, -1.632
vy_mid, vz_mid = 0.0, 0.0

N_int = 80  # snap integration samples

# Inequality enforcement (no negative thrust): a_z(t) <= grav - margin
M_samples = 60
margin = 1e-3

# ---------------- Drag parameters (linear, inertial frame) ----------------
rho_x = 0.6
rho_y = 0.6
rho_z = 0.4
# (these are linear coefficients so f_drag_i = -rho_i * v_i)

# ---------------- Helper functions ----------------
def time_power_vec(t, n):
    return np.array([t**i for i in range(n)], dtype=float)

def vel_basis(t, n):
    return np.array([0.0 if i==0 else i*t**(i-1) for i in range(n)], dtype=float)

def acc_basis(t, n):
    return np.array([0.0 if i<=1 else i*(i-1)*t**(i-2) for i in range(n)], dtype=float)

def jerk_basis(t, n):
    return np.array([0.0 if i<=2 else i*(i-1)*(i-2)*t**(i-3) for i in range(n)], dtype=float)

def eval_poly_np(coeffs, t):
    return sum(coeffs[i]*t**i for i in range(len(coeffs)))

def eval_vel_np(coeffs, t):
    return sum(i*coeffs[i]*t**(i-1) for i in range(1,len(coeffs)))

def eval_acc_np(coeffs, t):
    return sum(i*(i-1)*coeffs[i]*t**(i-2) for i in range(2,len(coeffs)))

def eval_jerk_np(coeffs, t):
    return sum(i*(i-1)*(i-2)*coeffs[i]*t**(i-3) for i in range(3,len(coeffs)))

def fit_initial_coeffs(constraints, n):
    A, b = [], []
    for typ, tval, val in constraints:
        if typ=='pos': A.append(time_power_vec(tval,n)); b.append(val)
        if typ=='vel': A.append(vel_basis(tval,n)); b.append(val)
        if typ=='acc': A.append(acc_basis(tval,n)); b.append(val)
        if typ=='jerk': A.append(jerk_basis(tval,n)); b.append(val)
    if len(A)==0:
        return np.zeros(n)
    sol, *_ = np.linalg.lstsq(np.vstack(A), np.array(b), rcond=None)
    if len(sol)<n:
        sol = np.concatenate([sol, np.zeros(n-len(sol))])
    return sol

# ---------------- CasADi NLP ----------------
cx = ca.SX.sym('cx', n)
cy = ca.SX.sym('cy', n)
cz = ca.SX.sym('cz', n)

# midpoint orientation & rate variables (decision vars)
u_sym  = ca.SX.sym('u')    # thrust magnitude at t_mid (T/m)
th_sym = ca.SX.sym('th')   # pitch at t_mid
ph_sym = ca.SX.sym('ph')   # roll at t_mid

ud_sym = ca.SX.sym('ud')   # u_dot at t_mid
thd_sym= ca.SX.sym('thd')  # theta_dot at t_mid
phd_sym= ca.SX.sym('phd')  # phi_dot at t_mid

t = ca.SX.sym('t')

# snap for cost
snap_x = sum([i*(i-1)*(i-2)*(i-3)*cx[i]*t**(i-4) for i in range(4,n)])
snap_y = sum([i*(i-1)*(i-2)*(i-3)*cy[i]*t**(i-4) for i in range(4,n)])
snap_z = sum([i*(i-1)*(i-2)*(i-3)*cz[i]*t**(i-4) for i in range(4,n)])

cost = 0
dt = (tf-t0)/float(N_int)
for i in range(N_int+1):
    ti = t0 + i*dt
    cost += ca.substitute(snap_x**2 + snap_y**2 + snap_z**2, t, ti) * dt

# equality/inequality constraint containers
g_cons = []
lbg = []
ubg = []

# start: pos, vel, acc  (equalities)
for ci, val in zip([cx, cy, cz], [x_start, y_start, z_start]):
    g_cons.append(ca.dot(ci, ca.DM(time_power_vec(t0,n))))
    lbg.append(float(val)); ubg.append(float(val))
    g_cons.append(ca.dot(ci, ca.DM(vel_basis(t0,n))))
    lbg.append(float(v0)); ubg.append(float(v0))
    g_cons.append(ca.dot(ci, ca.DM(acc_basis(t0,n))))
    lbg.append(float(a0)); ubg.append(float(a0))

# end: pos, vel, acc (equalities)
for ci, val in zip([cx, cy, cz], [x_end, y_end, z_end]):
    g_cons.append(ca.dot(ci, ca.DM(time_power_vec(tf,n))))
    lbg.append(float(val)); ubg.append(float(val))
    g_cons.append(ca.dot(ci, ca.DM(vel_basis(tf,n))))
    lbg.append(float(vf)); ubg.append(float(vf))
    g_cons.append(ca.dot(ci, ca.DM(acc_basis(tf,n))))
    lbg.append(float(af)); ubg.append(float(af))

# midpoint: position (equalities)
g_cons.append(ca.dot(cx, ca.DM(time_power_vec(t_mid,n))))
lbg.append(float(x_mid)); ubg.append(float(x_mid))
g_cons.append(ca.dot(cy, ca.DM(time_power_vec(t_mid,n))))
lbg.append(float(y_mid)); ubg.append(float(y_mid))
g_cons.append(ca.dot(cz, ca.DM(time_power_vec(t_mid,n))))
lbg.append(float(z_mid)); ubg.append(float(z_mid))

# midpoint: velocity (equalities)
g_cons.append(ca.dot(cy, ca.DM(vel_basis(t_mid,n))))
lbg.append(float(vy_mid)); ubg.append(float(vy_mid))
g_cons.append(ca.dot(cz, ca.DM(vel_basis(t_mid,n))))
lbg.append(float(vz_mid)); ubg.append(float(vz_mid))

# accelerations at midpoint (symbolic)
ax = ca.dot(cx, ca.DM(acc_basis(t_mid,n)))
ay = ca.dot(cy, ca.DM(acc_basis(t_mid,n)))
az = ca.dot(cz, ca.DM(acc_basis(t_mid,n)))

# --- LINEAR DRAG (symbolic) ---
# velocity at midpoint (symbolic)
vx_mid = ca.dot(cx, ca.DM(vel_basis(t_mid,n)))
vy_mid = ca.dot(cy, ca.DM(vel_basis(t_mid,n)))
vz_mid = ca.dot(cz, ca.DM(vel_basis(t_mid,n)))


# # drag per-axis: f_drag_i = -rho_i * v_i
fdx = ca.DM(rho_x) * vx_mid
fdy = ca.DM(rho_y) * vy_mid
fdz = ca.DM(rho_z) * vz_mid

# fdx = -ca.DM(rho_x) * ca.fabs(vx_mid) * vx_mid
# fdy = -ca.DM(rho_y) * ca.fabs(vy_mid) * vy_mid
# fdz = -ca.DM(rho_z) * ca.fabs(vz_mid) * vz_mid

# # fdot for linear drag: d/dt(-rho * v) = -rho * a
fdot_x = ca.DM(rho_x) * ax
fdot_y = ca.DM(rho_y) * ay
fdot_z = ca.DM(rho_z) * az

# fdot_x = -2 * ca.DM(rho_x) * ca.fabs(vx_mid) * ax
# fdot_y = -2 * ca.DM(rho_y) * ca.fabs(vy_mid) * ay
# fdot_z = -2 * ca.DM(rho_z) * ca.fabs(vz_mid) * az

# CORRECT NED mapping with linear drag: b = g*e3 - a + f_drag
# therefore components:
# bx = -ax + fdx
# by = -ay + fdy
# bz = g - az + fdz
# nonlinear attitude equalities (NED): b = u * R e3  (yaw=0)
eq1 = (-ax - fdx) - u_sym*ca.sin(th_sym)*ca.cos(ph_sym)
eq2 = (-ay - fdy) - u_sym*ca.sin(ph_sym)
eq3 = ca.DM(grav) - az - fdz - u_sym*ca.cos(ph_sym)*ca.cos(th_sym)

g_cons.extend([eq1, eq2, eq3])
lbg.extend([0.0, 0.0, 0.0])
ubg.extend([0.0, 0.0, 0.0])

# --- FORCE roll and pitch at midpoint (equalities)
g_cons.append(ph_sym); lbg.append(float(phi_des)); ubg.append(float(phi_des))
g_cons.append(th_sym); lbg.append(float(theta_des)); ubg.append(float(theta_des))

# --- derivative-level relation: jerk -> [ud, thd, phd]
# With linear drag: b_dot = -a_dot + fdot = jx_poly + fdot_x ...
jx = -ca.dot(cx, ca.DM(jerk_basis(t_mid,n)))
jy = -ca.dot(cy, ca.DM(jerk_basis(t_mid,n)))
jz = -ca.dot(cz, ca.DM(jerk_basis(t_mid,n)))

# total b_dot components:
bx_dot = jx - fdot_x
by_dot = jy - fdot_y
bz_dot = jz - fdot_z

s_th = ca.sin(th_sym); c_th = ca.cos(th_sym)
s_ph = ca.sin(ph_sym); c_ph = ca.cos(ph_sym)

A11 = s_th * c_ph
A12 = u_sym * c_th * c_ph
A13 = -u_sym * s_th * s_ph

A21 = -s_ph
A22 = 0
A23 = -u_sym * c_ph

A31 = c_th * c_ph
A32 = -u_sym * c_ph * s_th
A33 = -u_sym * s_ph * c_th

# A * [ud, thd, phd]^T = b_dot  (equalities)
g_cons.extend([
    A11*ud_sym + A12*thd_sym + A13*phd_sym - bx_dot,
    A21*ud_sym + A22*thd_sym + A23*phd_sym - by_dot,
    A31*ud_sym + A32*thd_sym + A33*phd_sym - bz_dot
])
lbg.extend([0.0, 0.0, 0.0])
ubg.extend([0.0, 0.0, 0.0])

# Force Euler-rate decision variables at midpoint to desired values (equalities)
g_cons.append(phd_sym); lbg.append(float(phi_dot_des)); ubg.append(float(phi_dot_des))
g_cons.append(thd_sym); lbg.append(float(theta_dot_des)); ubg.append(float(theta_dot_des))

for ti in [t0, tf]:
    jb = jerk_basis(float(ti), n)           # numeric numpy array
    jx_t = -ca.dot(cx, ca.DM(jb))           # -dot(coeffs, jerk_basis(ti))
    jb = jerk_basis(float(ti), n)
    jy_t = -ca.dot(cy, ca.DM(jb))
    jb = jerk_basis(float(ti), n)
    jz_t = -ca.dot(cz, ca.DM(jb))
    g_cons.extend([jx_t, jy_t, jz_t])
    lbg.extend([0.0, 0.0, 0.0])
    ubg.extend([0.0, 0.0, 0.0])

# ---------- add sampled inequality constraints: a_z(t_i) <= grav - margin ----------
for i in range(M_samples):
    ti = t0 + (i / float(max(1, M_samples - 1))) * (tf - t0)
    az_ti = ca.dot(cz, ca.DM(acc_basis(ti, n)))
    g_cons.append(az_ti)
    lbg.append(-ca.inf)                     # no lower bound
    ubg.append(float(grav - margin))        # upper bound: a_z <= grav - margin

# Decision vars and bounds
vars_all = ca.vertcat(cx, cy, cz, u_sym, th_sym, ph_sym, ud_sym, thd_sym, phd_sym)
lbx = [-ca.inf]*(3*n) + [1e-6, -np.pi/2, -np.pi/2, -100.0, -50.0, -50.0]
ubx = [ ca.inf]*(3*n) + [1e3,    np.pi/2,  np.pi/2,  100.0,  50.0,  50.0]

nlp = {'x': vars_all, 'f': cost, 'g': ca.vertcat(*g_cons)}
opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.max_iter': 2000}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# initial guess (coeffs + midpoint vars: u,theta,phi, ud, thd, phd)
x0 = np.concatenate([
    fit_initial_coeffs([('pos', t0, x_start), ('pos', t_mid, x_mid), ('pos', tf, x_end)], n),
    fit_initial_coeffs([('pos', t0, y_start), ('pos', t_mid, y_mid), ('pos', tf, y_end)], n),
    fit_initial_coeffs([('pos', t0, z_start), ('pos', t_mid, z_mid), ('pos', tf, z_end)], n),
    np.array([grav, theta0, phi_des, 0.0, theta_dot_des, phi_dot_des])
])

print("Solving NLP ...")
sol = solver(x0=x0.tolist(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
solx = sol['x'].full().flatten()

# extract solution
idx = 0
cx_sol = solx[idx: idx+n]; idx += n
cy_sol = solx[idx: idx+n]; idx += n
cz_sol = solx[idx: idx+n]; idx += n
u_sol = solx[idx]; th_sol = solx[idx+1]; ph_sol = solx[idx+2]; idx += 3
ud_sol = solx[idx]; thd_sol = solx[idx+1]; phd_sol = solx[idx+2]

print("Solved midpoint:")
print(" u    = {:.6f}, theta = {:.6f} deg, phi   = {:.6f} deg".format(u_sol, np.rad2deg(th_sol), np.rad2deg(ph_sol)))
print(" ud   = {:.6f}, thd   = {:.6f}, phd   = {:.6f}".format(ud_sol, thd_sol, phd_sol))

# Save coefficients
traj = {
    'x_axis': {'segment_1': cx_sol.tolist()},
    'y_axis': {'segment_1': cy_sol.tolist()},
    'z_axis': {'segment_1': cz_sol.tolist()},
    'timing': {'t0': t0, 'tf': tf, 't_mid': t_mid},
    'drag_rho': {'rho_x': rho_x, 'rho_y': rho_y, 'rho_z': rho_z}
}
with open("polynomial_trajectory.yaml", "w") as f:
    yaml.dump(traj, f, default_flow_style=False)
print("Saved coefficients to polynomial_trajectory.yaml")

# ---------- Evaluate trajectory & derivatives ----------
ts = np.linspace(t0, tf, 600)
x_vals = np.array([eval_poly_np(cx_sol, tt) for tt in ts])
y_vals = np.array([eval_poly_np(cy_sol, tt) for tt in ts])
z_vals = np.array([eval_poly_np(cz_sol, tt) for tt in ts])

vx = np.array([eval_vel_np(cx_sol, tt) for tt in ts])
vy = np.array([eval_vel_np(cy_sol, tt) for tt in ts])
vz = np.array([eval_vel_np(cz_sol, tt) for tt in ts])

ax_vals = np.array([eval_acc_np(cx_sol, tt) for tt in ts])
ay_vals = np.array([eval_acc_np(cy_sol, tt) for tt in ts])
az_vals = np.array([eval_acc_np(cz_sol, tt) for tt in ts])

jx_vals = np.array([eval_jerk_np(cx_sol, tt) for tt in ts])
jy_vals = np.array([eval_jerk_np(cy_sol, tt) for tt in ts])
jz_vals = np.array([eval_jerk_np(cz_sol, tt) for tt in ts])

# # ---------- compute linear drag and fdot (numerical) ----------
fdx_vals = rho_x * vx
fdy_vals = rho_y * vy
fdz_vals = rho_z * vz

# fdx_vals = -rho_x * np.abs(vx) * vx
# fdy_vals = -rho_y * np.abs(vy) * vy
# fdz_vals = -rho_z * np.abs(vz) * vz

# # fdot (linear): -rho * a
fdot_x_vals = rho_x * ax_vals
fdot_y_vals = rho_y * ay_vals
fdot_z_vals = rho_z * az_vals

# fdot_x_vals = -2 * rho_x * np.abs(vx) * ax_vals
# fdot_y_vals = -2 * rho_y * np.abs(vy) * ay_vals
# fdot_z_vals = -2 * rho_z * np.abs(vz) * az_vals

# CORRECT b = g*e3 - a + f_drag  (NED mapping with linear drag)
bx_vals = -ax_vals - fdx_vals
by_vals = -ay_vals - fdy_vals
bz_vals = grav - az_vals - fdz_vals

u_vals = np.sqrt(bx_vals**2 + by_vals**2 + bz_vals**2)
eps = 1e-12

# ---------- FIX: roll must be extracted from b_y/u (not from a_y/u) when drag is present ----------
# phi = asin( b_y / u ) where b_y = -ay + fdy
phi_vals = np.arcsin(np.clip(by_vals / (u_vals + eps), -1.0, 1.0))
# theta from atan2(bx, bz)
theta_vals = np.arctan2(bx_vals, bz_vals)

# compute Euler-rate time series by solving A(t) * rates = b_dot(t) = -jerk + fdot
u_dot_vals = np.zeros_like(ts)
theta_dot_vals = np.zeros_like(ts)
phi_dot_vals = np.zeros_like(ts)

for k, tt in enumerate(ts):
    u = u_vals[k]
    phi = phi_vals[k]
    theta = theta_vals[k]
    # b_dot = -jerk + fdot
    j = np.array([
        -jx_vals[k] - fdot_x_vals[k],
        -jy_vals[k] - fdot_y_vals[k],
        -jz_vals[k] - fdot_z_vals[k]
    ], dtype=float)

    s_th = np.sin(theta); c_th = np.cos(theta)
    s_ph = np.sin(phi); c_ph = np.cos(phi)

    A = np.array([
        [ s_th * c_ph,    u * c_th * c_ph,    -u * s_th * s_ph ],
        [ -s_ph,          0.0,                -u * c_ph         ],
        [ c_th * c_ph,   -u * c_ph * s_th,    -u * s_ph * c_th  ]
    ], dtype=float)

    # robust solve (least squares)
    sol_rates = np.linalg.lstsq(A, j, rcond=None)[0]
    u_dot_vals[k] = sol_rates[0]
    theta_dot_vals[k] = sol_rates[1]
    phi_dot_vals[k] = sol_rates[2]

# convert to degrees for plotting clarity
phi_deg_vals = np.rad2deg(phi_vals)
theta_deg_vals = np.rad2deg(theta_vals)
phi_dot_deg = np.rad2deg(phi_dot_vals)
theta_dot_deg = np.rad2deg(theta_dot_vals)

# ---------- Plots ----------
fig, axs = plt.subplots(2,2,figsize=(12,8))
axs[0,0].plot(ts, vx, label='vx'); axs[0,0].plot(ts, vy, label='vy'); axs[0,0].plot(ts, vz, label='vz')
axs[0,0].set_title('Velocity'); axs[0,0].legend(); axs[0,0].grid(True)

axs[0,1].plot(ts, ax_vals, label='ax'); axs[0,1].plot(ts, ay_vals, label='ay'); axs[0,1].plot(ts, az_vals, label='az')
axs[0,1].set_title('Acceleration (polynomial)'); axs[0,1].legend(); axs[0,1].grid(True)

axs[1,0].plot(ts, phi_deg_vals, label='roll (deg)'); axs[1,0].plot(ts, theta_deg_vals, label='pitch (deg)')
axs[1,0].axvline(t_mid, color='k', linestyle=':', linewidth=0.8)
axs[1,0].set_title('Orientations'); axs[1,0].set_xlabel('Time [s]'); axs[1,0].set_ylabel('deg'); axs[1,0].legend(); axs[1,0].grid(True)

axs[1,1].plot(ts, u_vals, label='u (||g*e3 - a + fdrag||)'); axs[1,1].axhline(grav, color='k', linestyle=':', linewidth=0.8, label='g'); axs[1,1].axvline(t_mid, color='k', linestyle=':', linewidth=0.8)
axs[1,1].set_title('u (acc norm w/ linear drag)'); axs[1,1].set_xlabel('Time [s]'); axs[1,1].set_ylabel('m/s^2'); axs[1,1].legend(); axs[1,1].grid(True)
plt.tight_layout()
plt.show(block=False)

# angular rates + u_dot plot
fig2, axs2 = plt.subplots(2,1,figsize=(10,6), sharex=True)
axs2[0].plot(ts, phi_dot_deg, label='roll rate (deg/s)'); axs2[0].plot(ts, theta_dot_deg, label='pitch rate (deg/s)')
axs2[0].axvline(t_mid, color='k', linestyle=':', linewidth=0.8); axs2[0].set_title('Euler rates (from jerk + linear fdot)'); axs2[0].legend(); axs2[0].grid(True)
axs2[1].plot(ts, u_dot_vals, label='u_dot (m/s^3)'); axs2[1].axvline(t_mid, color='k', linestyle=':', linewidth=0.8); axs2[1].set_xlabel('Time [s]'); axs2[1].legend(); axs2[1].grid(True)
plt.tight_layout()
plt.show(block=False)

# 3D trajectory
fig3 = plt.figure(figsize=(8,6)); ax3d = fig3.add_subplot(111, projection='3d')
ax3d.plot(x_vals, y_vals, z_vals, label='traj'); ax3d.scatter([x_start,x_mid,x_end],[y_start,y_mid,y_end],[z_start,z_mid,z_end], color='red', label='constraints')
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z'); ax3d.legend(); plt.tight_layout(); plt.show()

# ---------- verification at t_mid ----------
ax_m = eval_acc_np(cx_sol, t_mid)
ay_m = eval_acc_np(cy_sol, t_mid)
az_m = eval_acc_np(cz_sol, t_mid)
jx_m = eval_jerk_np(cx_sol, t_mid)
jy_m = eval_jerk_np(cy_sol, t_mid)
jz_m = eval_jerk_np(cz_sol, t_mid)

vx_m = eval_vel_np(cx_sol, t_mid)
vy_m = eval_vel_np(cy_sol, t_mid)
vz_m = eval_vel_np(cz_sol, t_mid)

fdx_m = -rho_x * vx_m
fdy_m = -rho_y * vy_m
fdz_m = -rho_z * vz_m

# fdx_m = -rho_x * abs(vx_m) * vx_m
# fdy_m = -rho_y * abs(vy_m) * vy_m
# fdz_m = -rho_z * abs(vz_m) * vz_m

bx_m = -ax_m + fdx_m
by_m = -ay_m + fdy_m
bz_m = grav - az_m + fdz_m
u_m = np.sqrt(bx_m**2 + by_m**2 + bz_m**2)

# ---------- FIX: use b_y/u for phi (including drag) ----------
phi_m = np.arcsin(np.clip(by_m / (u_m + 1e-12), -1.0, 1.0))
theta_m = np.arctan2(bx_m, bz_m)

# fdot at midpoint (numerical)
fdot_x_m = -rho_x * ax_m
fdot_y_m = -rho_y * ay_m
fdot_z_m = -rho_z * az_m

A_mid = np.array([
    [ np.sin(theta_m)*np.cos(phi_m),  u_m*np.cos(theta_m)*np.cos(phi_m),  -u_m*np.sin(theta_m)*np.sin(phi_m) ],
    [ -np.sin(phi_m),                 0.0,                                 -u_m*np.cos(phi_m) ],
    [ np.cos(theta_m)*np.cos(phi_m), -u_m*np.cos(phi_m)*np.sin(theta_m),  -u_m*np.sin(phi_m)*np.cos(theta_m) ]
], dtype=float)
j_mid = np.array([
    -jx_m + fdot_x_m,
    -jy_m + fdot_y_m,
    -jz_m + fdot_z_m
], dtype=float)  # note b_dot = -a_dot + fdot
rates_mid = np.linalg.lstsq(A_mid, j_mid, rcond=None)[0]

print("\n--- checks at t_mid ---")
print("requested phi_des (deg)      =", phi_deg)
print("ph_sym (solver)   deg        =", np.rad2deg(ph_sol))
print("phi from b (deg)              =", np.rad2deg(phi_m))
print("theta from b (deg)            =", np.rad2deg(theta_m))
print("u (solver) / u (from b)       = {:.6f} / {:.6f}".format(u_sol, u_m))
print("requested phd (deg/s)         =", np.rad2deg(phi_dot_des))
print("requested thd (deg/s)         =", np.rad2deg(theta_dot_des))
print("phd_sym (solver)              =", phd_sol)
print("thd_sym (solver)              =", thd_sol)
print("rates from A_mid \\ j_mid     =", rates_mid)
