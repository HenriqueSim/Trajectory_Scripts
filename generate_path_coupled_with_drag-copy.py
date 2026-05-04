#!/usr/bin/env python3
"""
Single-segment polynomial trajectory with:
 - midpoint constraints: position, roll (phi) and Euler-rates (phi_dot, theta_dot)
 - exact nonlinear coupling (NED) via R matrix
 - derivative-level (jerk -> Euler-rates) exact linear relation enforced at midpoint
 - minimize integrated snap^2
 - enforce a_z(t) <= g - margin at M_samples times (no 'negative thrust' requirement)
Produces plots for velocities, accelerations, orientations, u and angular rates.

NLP now builds R = Rz(psi) * Ry(theta) * Rx(phi) and enforces:
  R[:,2] == Z_b_des (Z_b_des = -b / ||b||), u == ||b||
  body-rates (p,q,r) from Euler rates constrained to controller w_des computed from jerk:
    w0 = (1/||b||) * Y_b.dot(jerk)
    w1 = -(1/||b||) * X_b.dot(jerk)
    w2 = yaw_rate * Z_b[2]
"""
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import yaml

# ---------------- User parameters ----------------
order = 20
n = order + 1
t0 = 0.0
tf = 3.5
t_mid = 1.75
grav = 9.81

phi_deg = 20.0
phi_des = np.deg2rad(phi_deg)   # desired roll at midpoint (for reference)
theta_deg_des = 0.0
theta_des = np.deg2rad(theta_deg_des)  # desired pitch at midpoint (for reference)
psi_des = 0.0                   # yaw fixed
theta0 = 0.0                    # initial pitch guess

# Desired Euler-rate at midpoint (rad/s). (kept for reference)
phi_dot_des = 0.0
theta_dot_des = 0.0

# Start/end positions
x_start, x_end = 2.0, -2.0
y_start, y_end = 0.0, 0.0
z_start, z_end = -1.3, -1.3

# Start/end velocities, accelerations and jerks
v0, vf = 0.0, 0.0
a0, af = 0.0, 0.0

# Midpoint (position)
x_mid, y_mid, z_mid = 0.0, 0.0, -1.8

N_int = 80  # snap integration samples

# Inequality enforcement (no negative thrust): a_z(t) <= grav - margin
M_samples = 60
margin = 1e-3

# ---------------- Drag parameters (linear, inertial frame) ----------------
rho_x = 0.560
rho_y = 0.565
rho_z = 0.0
# (these are linear coefficients so f_drag_i = -rho_i * v_i)

# ---------------- Additional params to match C++ controller snippet ----------
mass = 1.0            # mass cancels out in the formula but kept for clarity
yaw_rate_rad = 0.0    # desired yaw rate

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
u_sym   = ca.SX.sym('u')     # thrust magnitude at t_mid (T/m) -> constrained to ||b||
th_sym  = ca.SX.sym('th')    # pitch at t_mid (theta)
ph_sym  = ca.SX.sym('ph')    # roll at t_mid (phi)
psi_sym = ca.SX.sym('psi')   # yaw at t_mid (psi) - fixed by equality to psi_des

ud_sym  = ca.SX.sym('ud')    # u_dot at t_mid
thd_sym = ca.SX.sym('thd')   # theta_dot at t_mid
phd_sym = ca.SX.sym('phd')   # phi_dot at t_mid
psd_sym = ca.SX.sym('psd')   # psi_dot at t_mid

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

# accelerations at midpoint (symbolic)
ax = ca.dot(cx, ca.DM(acc_basis(t_mid,n)))
ay = ca.dot(cy, ca.DM(acc_basis(t_mid,n)))
az = ca.dot(cz, ca.DM(acc_basis(t_mid,n)))

# --- LINEAR DRAG (symbolic) kept for other parts of the problem (not used in new R-based equalities) ---
vx_mid = ca.dot(cx, ca.DM(vel_basis(t_mid,n)))
vy_mid = ca.dot(cy, ca.DM(vel_basis(t_mid,n)))
vz_mid = ca.dot(cz, ca.DM(vel_basis(t_mid,n)))

fdx = ca.DM(rho_x) * vx_mid
fdy = ca.DM(rho_y) * vy_mid
fdz = ca.DM(rho_z) * vz_mid

fdot_x = ca.DM(rho_x) * ax
fdot_y = ca.DM(rho_y) * ay
fdot_z = ca.DM(rho_z) * az

# -------------------- NEW: R-matrix based equalities --------------------
eps = 1e-12

# build b and its norm (controller convention)
b_vec = ca.vertcat(ax, ay, az) - ca.vertcat(0.0, 0.0, grav)   # b = a - [0,0,g]
b_norm = ca.sqrt(ca.dot(b_vec, b_vec))
b_norm_safe = b_norm + eps

# ensure u_sym equals ||b||
g_cons.append(u_sym - b_norm)
lbg.append(0.0); ubg.append(0.0)

# Build R = Rz(psi) * Ry(theta) * Rx(phi)
# Rx(phi)
c_p = ca.cos(ph_sym); s_p = ca.sin(ph_sym)
Rx = ca.vertcat(
    ca.horzcat(1.0, 0.0, 0.0),
    ca.horzcat(0.0, c_p, -s_p),
    ca.horzcat(0.0, s_p,  c_p)
)

# Ry(theta)
c_t = ca.cos(th_sym); s_t = ca.sin(th_sym)
Ry = ca.vertcat(
    ca.horzcat(c_t, 0.0, s_t),
    ca.horzcat(0.0, 1.0, 0.0),
    ca.horzcat(-s_t,0.0, c_t)
)

# Rz(psi)
c_y = ca.cos(psi_sym); s_y = ca.sin(psi_sym)
Rz = ca.vertcat(
    ca.horzcat(c_y, -s_y, 0.0),
    ca.horzcat(s_y,  c_y, 0.0),
    ca.horzcat(0.0,  0.0, 1.0)
)

R = ca.mtimes(Rz, ca.mtimes(Ry, Rx))  # 3x3 symbolic rotation matrix

# Enforce R[:,2] == Z_b_des = -b / ||b||
Z_b_des = - b_vec / b_norm_safe
# R third column equals Z_b_des -> 3 equalities
g_cons.append(R[0,2] - Z_b_des[0]); lbg.append(0.0); ubg.append(0.0)
g_cons.append(R[1,2] - Z_b_des[1]); lbg.append(0.0); ubg.append(0.0)
g_cons.append(R[2,2] - Z_b_des[2]); lbg.append(0.0); ubg.append(0.0)

# Optionally fix yaw to psi_des (keeps yaw locked; change if you want yaw free)
g_cons.append(psi_sym - float(psi_des)); lbg.append(0.0); ubg.append(0.0)

# --- jerk at midpoint (symbolic) ---
jx = -ca.dot(cx, ca.DM(jerk_basis(t_mid,n)))
jy = -ca.dot(cy, ca.DM(jerk_basis(t_mid,n)))
jz = -ca.dot(cz, ca.DM(jerk_basis(t_mid,n)))
jerk_vec = ca.vertcat(jx, jy, jz)

# X_b = R[:,0], Y_b = R[:,1], Z_b = R[:,2]
X_b = ca.vertcat(R[0,0], R[1,0], R[2,0])
Y_b = ca.vertcat(R[0,1], R[1,1], R[2,1])
Z_b = ca.vertcat(R[0,2], R[1,2], R[2,2])

# controller desired body rates from jerk (mass/T -> 1/||b||)
w0 = (1.0 / b_norm_safe) * ca.dot(Y_b, jerk_vec)   # p desired
w1 = -(1.0 / b_norm_safe) * ca.dot(X_b, jerk_vec)  # q desired
w2 = ca.DM(yaw_rate_rad) * Z_b[2]                  # r desired (body z component scaled by yaw_rate)

# Map Euler rates (phd_sym, thd_sym, psd_sym) -> body rates (p,q,r) for ZYX euler sequence:
# p = phi_dot - psi_dot * sin(theta)
# q = cos(phi)*theta_dot + sin(phi)*cos(theta)*psi_dot
# r = -sin(phi)*theta_dot + cos(phi)*cos(theta)*psi_dot
p_from_euler = phd_sym - psd_sym * ca.sin(th_sym)
q_from_euler = ca.cos(ph_sym) * thd_sym + ca.sin(ph_sym) * ca.cos(th_sym) * psd_sym
r_from_euler = -ca.sin(ph_sym) * thd_sym + ca.cos(ph_sym) * ca.cos(th_sym) * psd_sym

# enforce body rate equalities: (p,q,r) == (w0,w1,w2)
g_cons.append(p_from_euler - w0); lbg.append(0.0); ubg.append(0.0)
g_cons.append(q_from_euler - w1); lbg.append(0.0); ubg.append(0.0)
g_cons.append(r_from_euler - w2); lbg.append(0.0); ubg.append(0.0)

# (Removed old A*[ud,thd,phd] = b_dot block and old forced phd/thd equalities)

# add jerk == 0 at endpoints (kept from original)
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

# Decision vars and bounds (added psi_sym, psd_sym)
vars_all = ca.vertcat(cx, cy, cz, u_sym, th_sym, ph_sym, psi_sym, ud_sym, thd_sym, phd_sym, psd_sym)
lbx = [-ca.inf]*(3*n) + [1e-6, -np.pi/2, -np.pi/2, -np.pi, -100.0, -100.0, -100.0, -100.0]
ubx = [ ca.inf]*(3*n) + [1e3,    np.pi/2,  np.pi/2,  np.pi,  100.0,  100.0,  100.0,  100.0]

# Note: lbx/ubx appended: u, th, ph, psi, ud, thd, phd, psd
# adjust sizes if you change ordering

nlp = {'x': vars_all, 'f': cost, 'g': ca.vertcat(*g_cons)}
opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.max_iter': 2000}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# initial guess (coeffs + midpoint vars: u,theta,phi,psi, ud, thd, phd, psd)
x0 = np.concatenate([
    fit_initial_coeffs([('pos', t0, x_start), ('pos', t_mid, x_mid), ('pos', tf, x_end)], n),
    fit_initial_coeffs([('pos', t0, y_start), ('pos', t_mid, y_mid), ('pos', tf, y_end)], n),
    fit_initial_coeffs([('pos', t0, z_start), ('pos', t_mid, z_mid), ('pos', tf, z_end)], n),
    np.array([grav, theta0, phi_des, psi_des, 0.0, theta_dot_des, phi_dot_des, 0.0])
])

print("Solving NLP ...")
sol = solver(x0=x0.tolist(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
solx = sol['x'].full().flatten()

# --- diagnostics (paste right after solx = ...) ---
solx_ca = ca.DM(solx)  # cast to CasADi DM

# solver stats (some solvers provide 'status' via solver.stats())
try:
    stats = solver.stats()
except Exception:
    stats = {}
print("\n--- solver stats ---")
print(" solver return:", sol.keys())
print(" solver stats:", stats)

# evaluate all constraints g_cons compactly
G = ca.vertcat(*g_cons)
G_fun = ca.Function('G_fun', [vars_all], [G])
g_eval = G_fun(solx_ca).full().flatten()
print(" #constraints:", g_eval.size)
print(" max |g|    =", np.max(np.abs(g_eval)))
print(" mean |g|   =", np.mean(np.abs(g_eval)))
print(" g eval (first 40) =", g_eval[:40])

# Helpful per-block residuals — recompute known groups numerically:
# 1) boundary pos/vel/acc at t0, tf and midpoint pos
def eval_basis_group(coeffs, tval, n):
    return np.dot(coeffs, time_power_vec(tval, n))

# positions at t0, t_mid, tf
cx_num = solx[0:n]; cy_num = solx[n:2*n]; cz_num = solx[2*n:3*n]
print("\nBoundary checks:")
print(" x(t0) error =", eval_poly_np(cx_num, t0) - x_start)
print(" x(t_mid) err=", eval_poly_np(cx_num, t_mid) - x_mid)
print(" x(tf) error =", eval_poly_np(cx_num, tf) - x_end)
print(" z(t_mid) error=", eval_poly_np(cz_num, t_mid) - z_mid)

# 2) R[:,2] = -b/||b|| checks at midpoint
ax_m = eval_acc_np(cx_num, t_mid)
ay_m = eval_acc_np(cy_num, t_mid)
az_m = eval_acc_np(cz_num, t_mid)
b_m = np.array([ax_m, ay_m, az_m]) - np.array([0.0, 0.0, grav])
bnorm = np.linalg.norm(b_m) + 1e-12
Zb_num = -b_m / bnorm

# compute R from solution Euler angles
ph_sol_idx = 3*n + 1  # depends on your variable ordering; adjust if different
# safer: use extracted ph_sol, th_sol, psi_sol after you extract them below
print("\n R[:,2] vs controller Z_b (abs errors):")
# We'll compare numerically by reconstructing R from ph_sol,th_sol,psi_sol later (after extraction)

# 3) sampled a_z inequality violations (should be <= grav - margin)
violations = []
for i in range(M_samples):
    ti = t0 + (i / float(max(1, M_samples - 1))) * (tf - t0)
    az_ti = eval_acc_np(cz_num, ti)
    if az_ti > (grav - margin) + 1e-9:
        violations.append((i, ti, az_ti))
print(" #a_z inequality violations at sample points =", len(violations))
if len(violations) > 0:
    print(" first violations:", violations[:5])

# extract solution (respect ordering used above)
idx = 0
cx_sol = solx[idx: idx+n]; idx += n
cy_sol = solx[idx: idx+n]; idx += n
cz_sol = solx[idx: idx+n]; idx += n
u_sol = solx[idx]; th_sol = solx[idx+1]; ph_sol = solx[idx+2]; psi_sol = solx[idx+3]; idx += 4
ud_sol = solx[idx]; thd_sol = solx[idx+1]; phd_sol = solx[idx+2]; psd_sol = solx[idx+3]

print("Solved midpoint:")
print(" u    = {:.6f}, theta = {:.6f} deg, phi   = {:.6f} deg, psi = {:.6f} deg".format(u_sol, np.rad2deg(th_sol), np.rad2deg(ph_sol), np.rad2deg(psi_sol)))
print(" ud   = {:.6f}, thd   = {:.6f}, phd   = {:.6f}, psd = {:.6f}".format(ud_sol, thd_sol, phd_sol, psd_sol))

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

# ---------- Evaluate trajectory & derivatives (numeric time series) ----------
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

# (rest of script: compute numeric R-based attitudes and rates for plotting and verification)
# ---------- compute attitude/time-series using controller mapping (numerical) ----------
acc_vecs = np.vstack([ax_vals, ay_vals, az_vals]).T
jerk_vecs = np.vstack([jx_vals, jy_vals, jz_vals]).T

phi_vals = np.zeros_like(ts)
theta_vals = np.zeros_like(ts)
psi_vals = np.zeros_like(ts)
u_vals = np.zeros_like(ts)
u_dot_vals = np.zeros_like(ts)
phi_dot_vals = np.zeros_like(ts)
theta_dot_vals = np.zeros_like(ts)
psi_dot_vals = np.zeros_like(ts)

Y_C = np.array([0.0, 1.0, 0.0], dtype=float)
eps_np = 1e-12

for k, tt in enumerate(ts):
    a_vec = acc_vecs[k]
    j_vec = jerk_vecs[k]
    b_vec_np = a_vec - np.array([0.0, 0.0, grav], dtype=float)
    b_norm_np = np.linalg.norm(b_vec_np)
    if b_norm_np < 1e-8:
        Z_b = np.array([0.0, 0.0, 1.0], dtype=float)
        X_b = np.array([1.0, 0.0, 0.0], dtype=float)
        Y_b = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        Z_b = - b_vec_np / (b_norm_np + eps_np)
        X_b = np.cross(Y_C, Z_b)
        X_bn = np.linalg.norm(X_b)
        if X_bn < 1e-8:
            if abs(Z_b[2]) < 0.99:
                X_b = np.cross([0.0,0.0,1.0], Z_b)
            else:
                X_b = np.cross([1.0,0.0,0.0], Z_b)
            X_bn = np.linalg.norm(X_b)
        X_b = X_b / (X_bn + eps_np)
        Y_b = np.cross(Z_b, X_b)
        Y_b = Y_b / (np.linalg.norm(Y_b) + eps_np)

    # numeric R from columns
    R_num = np.column_stack([X_b, Y_b, Z_b])

    # extract Euler ZYX: pitch = asin(-R[2,0]) = asin(-X_b[2]), roll = atan2(R[2,1], R[2,2]) = atan2(Y_b[2], Z_b[2]), yaw = atan2(R[1,0], R[0,0]) = atan2(X_b[1], X_b[0])
    pitch = np.arcsin(np.clip(-X_b[2], -1.0, 1.0))
    roll  = np.arctan2(Y_b[2], Z_b[2])
    yaw   = np.arctan2(X_b[1], X_b[0])

    # body rates from jerk
    if b_norm_np < 1e-8:
        w0 = 0.0; w1 = 0.0
    else:
        w0 = (1.0 / (b_norm_np + eps_np)) * np.dot(Y_b, j_vec)
        w1 = -(1.0 / (b_norm_np + eps_np)) * np.dot(X_b, j_vec)
    w2 = yaw_rate_rad * Z_b[2]

    # now map Euler rates to body rates: we don't have actual phd/thd/psd time series here (only midpoint constrained), so we leave plotting based on controller w_des
    phi_vals[k] = roll
    theta_vals[k] = pitch
    psi_vals[k] = yaw
    phi_dot_vals[k] = w0
    theta_dot_vals[k] = w1
    psi_dot_vals[k] = w2
    u_vals[k] = b_norm_np

    # numerical u_dot
    if k==0:
        u_dot_vals[k] = (u_vals[k+1] - u_vals[k])/(ts[1]-ts[0])
    elif k==len(ts)-1:
        u_dot_vals[k] = (u_vals[k] - u_vals[k-1])/(ts[1]-ts[0])
    else:
        u_dot_vals[k] = (u_vals[k+1] - u_vals[k-1])/(ts[k+1]-ts[k-1])

# convert to degrees for plotting clarity
phi_deg_vals = np.rad2deg(phi_vals)
theta_deg_vals = np.rad2deg(theta_vals)
phi_dot_deg = np.rad2deg(phi_dot_vals)
theta_dot_deg = np.rad2deg(theta_dot_vals)

# ---------- Plots (same as before) ----------
fig, axs = plt.subplots(2,2,figsize=(12,8))
axs[0,0].plot(ts, vx, label='vx'); axs[0,0].plot(ts, vy, label='vy'); axs[0,0].plot(ts, vz, label='vz')
axs[0,0].set_title('Velocity'); axs[0,0].legend(); axs[0,0].grid(True)

axs[0,1].plot(ts, ax_vals, label='ax'); axs[0,1].plot(ts, ay_vals, label='ay'); axs[0,1].plot(ts, az_vals, label='az')
axs[0,1].set_title('Acceleration (polynomial)'); axs[0,1].legend(); axs[0,1].grid(True)

axs[1,0].plot(ts, phi_deg_vals, label='roll (deg)'); axs[1,0].plot(ts, theta_deg_vals, label='pitch (deg)')
axs[1,0].axvline(t_mid, color='k', linestyle=':', linewidth=0.8)
axs[1,0].set_title('Orientations (controller-style via R)'); axs[1,0].set_xlabel('Time [s]'); axs[1,0].set_ylabel('deg'); axs[1,0].legend(); axs[1,0].grid(True)

axs[1,1].plot(ts, u_vals, label='u (||b||) per controller'); axs[1,1].axhline(grav, color='k', linestyle=':', linewidth=0.8, label='g'); axs[1,1].axvline(t_mid, color='k', linestyle=':', linewidth=0.8)
axs[1,1].set_title('u (controller acc norm)'); axs[1,1].set_xlabel('Time [s]'); axs[1,1].set_ylabel('m/s^2'); axs[1,1].legend(); axs[1,1].grid(True)
plt.tight_layout()
plt.show(block=False)

# angular rates + u_dot plot
fig2, axs2 = plt.subplots(2,1,figsize=(10,6), sharex=True)
axs2[0].plot(ts, phi_dot_deg, label='roll rate (deg/s)'); axs2[0].plot(ts, theta_dot_deg, label='pitch rate (deg/s)')
axs2[0].axvline(t_mid, color='k', linestyle=':', linewidth=0.8); axs2[0].set_title('Euler rates (controller-style)'); axs2[0].legend(); axs2[0].grid(True)
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

# controller-style numeric check at midpoint
b_m = np.array([ax_m, ay_m, az_m]) - np.array([0.0, 0.0, grav])
b_norm_m = np.linalg.norm(b_m) + 1e-12
Z_b_m = -b_m / (b_norm_m)
# reconstruct X_b,Y_b robustly as earlier
X_b_m = np.cross(np.array([0.0,1.0,0.0]), Z_b_m)
if np.linalg.norm(X_b_m) < 1e-12:
    if abs(Z_b_m[2]) < 0.99:
        X_b_m = np.cross([0.0,0.0,1.0], Z_b_m)
    else:
        X_b_m = np.cross([1.0,0.0,0.0], Z_b_m)
X_b_m /= (np.linalg.norm(X_b_m) + 1e-12)
Y_b_m = np.cross(Z_b_m, X_b_m)
Y_b_m /= (np.linalg.norm(Y_b_m) + 1e-12)

R_des_m = np.column_stack([X_b_m, Y_b_m, Z_b_m])
# extract euler
if abs(R_des_m[2,0]) < 1.0 - 1e-12:
    pitch_m = np.arcsin(-R_des_m[2,0])
    roll_m  = np.arctan2(R_des_m[2,1], R_des_m[2,2])
    yaw_m   = np.arctan2(R_des_m[1,0], R_des_m[0,0])
else:
    pitch_m = np.pi/2 * np.sign(-R_des_m[2,0])
    roll_m  = 0.0
    yaw_m   = np.arctan2(-R_des_m[0,1], R_des_m[1,1])

j_mid = np.array([jx_m, jy_m, jz_m])
w_des_mid = np.array([
    (1.0 / b_norm_m) * np.dot(Y_b_m, j_mid),
    -(1.0 / b_norm_m) * np.dot(X_b_m, j_mid),
    yaw_rate_rad * Z_b_m[2]
])

print("\n--- controller-style checks at t_mid ---")
print("requested phi_des (deg)      =", phi_deg)
print("phi (from controller R) deg  =", np.rad2deg(roll_m))
print("theta (from controller R) deg=", np.rad2deg(pitch_m))
print("u (from controller b_norm)   =", b_norm_m)
print("controller w_des_mid (deg/s) =", np.rad2deg(w_des_mid))

# Keep old checks too (for comparison)
bx_m_old = -ax_m - (-rho_x * vx_m)
by_m_old = -ay_m - (-rho_y * vy_m)
bz_m_old = grav - az_m - (-rho_z * vz_m)
u_m_old = np.sqrt(bx_m_old**2 + by_m_old**2 + bz_m_old**2)

print("solver ph_sym (deg)           =", np.rad2deg(ph_sol))
print("u (solver) / u_old            = {:.6f} / {:.6f}".format(u_sol, u_m_old))
