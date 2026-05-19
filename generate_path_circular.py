#!/usr/bin/env python3
"""
Multi-segment circular trajectory generator.

Structure
---------
  Segment 0          : entry — played exactly ONCE.
                       Starts at HOVER_START (vel=acc=0).
                       Ends with C2 continuity into the first circular segment.

  Segments 1..N_CIRC : circular lap — repeated N_LAPS times.
                       C2 continuity at every internal junction AND at the
                       lap-wrap boundary (end of last circ → start of first circ).

Continuity: C2 (pos + vel + acc).  C4 was infeasible with inclined obstacles.

Cost: minimise integrated snap^2 per segment.

Inclined obstacles
------------------
  inclination_deg : angle wall makes with horizontal.
                    90° = vertical (no roll), 50° → phi=40°, 45° → phi=45°
                    Formula: phi_auto = 90 - inclination_deg
  inclination_axis: 'x' or 'y' — plane the wall tilts in
  approach_dir    : +1 or -1 — sign of v·n at passage (ALWAYS SET EXPLICITLY)
                    +1 if drone moves in +normal direction, -1 otherwise

  Quick reference for axis='y':
    inclination_deg  phi_des   use-case
    90               0°        vertical wall, no roll
    70               20°       gentle lean
    50               40°       steep lean
    45               45°       45° dive
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml

# ============================================================
#  USER PARAMETERS
# ============================================================

N_LAPS           = 2
N_ENTRY_SEGMENTS = 1
ORDER            = 11    # C2: 12 coefficients, 3 constraints/axis/junction
N_INT            = 80    # snap-cost integration samples per segment
M_SAMP           = 60    # a_z inequality samples per segment
MARGIN           = 1e-3
GRAV             = 9.81
RHO_X = RHO_Y = RHO_Z = 0.0

MIN_PASS_SPEED = 0.5   # minimum |v·n| at inclined window [m/s]

# ── Hard workspace box constraints ────────────────────────────────────────────
B_SAMP       = 30
X_MIN, X_MAX = -3.0,  3.0
Y_MIN, Y_MAX = -2.0,  2.0
Z_MAX        = -0.5   # drone stays above 0.5 m (NED: z < Z_MAX = z more negative)

HOVER_START = (2.0, -1.0, -1.5)

# ── Segments ──────────────────────────────────────────────────────────────────
SEGMENTS = [
    # ── Entry ─────────────────────────────────────────────────────────────────
    {
        'duration' : 2.5,
        'obstacle' : {
            'pos'       : (2.0, 0.0, -1.5),
            'phi_deg'   : 0.0,
            'theta_deg' : 0.0,
            'phi_dot'   : 0.0,
            'theta_dot' : 0.0,
            'vx'        : 0.0,
            'vy'        : None,
            'vz'        : 0.0,
        },
    },
    # ── Circ 1 ────────────────────────────────────────────────────────────────
    {
        'duration' : 2.5,
        'obstacle' : {
            'pos'       : (0.0, 1.0, -1.5),
            'phi_deg'   : 0.0,
            'theta_deg' : 0.0,
            'phi_dot'   : 0.0,
            'theta_dot' : 0.0,
            'vx'        : None,
            'vy'        : 0.0,
            'vz'        : 0.0,
        },
    },
    # ── Circ 2 — inclined window ───────────────────────────────────────────────
    # Wall normal in NED: n = (0, -sin(50°), +cos(50°)) = (0, -0.766, +0.643)
    # Drone moves in -y and +z (descending) to pass perpendicularly through it.
    # phi = 90-50 = 40° with yaw=0 generates the required -y thrust component.
    # approach_dir=-1: force vy ≤ -MIN_PASS_SPEED (drone moves in -y direction)
    # With tangent constraint: vz = -vy·cos(α)/sin(α) > 0 (descending) ✓
    {
        'duration' : 1.0,
        'obstacle' : {
            'pos'              : (-2.0, 0.0, -1.625),
            'phi_dot'          : 0.0,
            'theta_dot'        : 0.0,
            'vx'               : 0.0,
            'vy'               : None,
            'vz'               : None,
            'inclination_deg'  : 50.0,   # phi_auto = 90-50 = 40°
            'inclination_axis' : 'y',
            'approach_dir'     : -1,     # -1 = force vy < 0 (drone moves in -y)
        },
    },
    # ── Circ 3 ────────────────────────────────────────────────────────────────
    {
        'duration' : 3.0,
        'obstacle' : {
            'pos'       : (0.0, -1.0, -1.5),
            'phi_deg'   : 0.0,
            'theta_deg' : 0.0,
            'phi_dot'   : 0.0,
            'theta_dot' : 0.0,
            'vx'        : None,
            'vy'        : 0.0,
            'vz'        : 0.0,
        },
    },
    # ── Circ 4 ────────────────────────────────────────────────────────────────
    {
        'duration' : 2.5,
        'obstacle' : {
            'pos'       : (2.0, 0.0, -1.5),
            'phi_deg'   : 0.0,
            'theta_deg' : 0.0,
            'phi_dot'   : 0.0,
            'theta_dot' : 0.0,
            'vx'        : 0.0,
            'vy'        : None,
            'vz'        : 0.0,
        },
    },
]

# ============================================================
#  DERIVED CONSTANTS
# ============================================================

S       = len(SEGMENTS)
N_CIRC  = S - N_ENTRY_SEGMENTS
n       = ORDER + 1
first_c = N_ENTRY_SEGMENTS

entry_dur = sum(SEGMENTS[i]['duration'] for i in range(N_ENTRY_SEGMENTS))
circ_dur  = sum(SEGMENTS[i]['duration'] for i in range(N_ENTRY_SEGMENTS, S))
total_dur = entry_dur + N_LAPS * circ_dur

print(f"\nSegments: {N_ENTRY_SEGMENTS} entry + {N_CIRC} circular")
print(f"Entry: {entry_dur:.2f}s  |  Lap: {circ_dur:.2f}s  |  "
      f"Total ({N_LAPS} laps): {total_dur:.2f}s")
print(f"ORDER={ORDER}  n_coeffs={n}  vars={S*(3*n+6)}")

# ============================================================
#  POLYNOMIAL BASIS HELPERS
# ============================================================

def time_power_vec(t, n):
    return np.array([t**i for i in range(n)], dtype=float)

def vel_basis(t, n):
    return np.array([0.0 if i == 0 else i*t**(i-1) for i in range(n)], dtype=float)

def acc_basis(t, n):
    return np.array([0.0 if i <= 1 else i*(i-1)*t**(i-2) for i in range(n)], dtype=float)

def jerk_basis(t, n):
    return np.array([0.0 if i <= 2 else i*(i-1)*(i-2)*t**(i-3) for i in range(n)], dtype=float)

def eval_poly(c, t):  return sum(c[i]*t**i for i in range(len(c)))
def eval_vel(c, t):   return sum(i*c[i]*t**(i-1) for i in range(1, len(c)))
def eval_acc(c, t):   return sum(i*(i-1)*c[i]*t**(i-2) for i in range(2, len(c)))

def fit_initial_coeffs(constraints, n):
    bases = {'pos': time_power_vec, 'vel': vel_basis, 'acc': acc_basis}
    A, b = [], []
    for typ, tval, val in constraints:
        A.append(bases[typ](tval, n)); b.append(val)
    if not A: return np.zeros(n)
    sol, *_ = np.linalg.lstsq(np.vstack(A), np.array(b), rcond=None)
    return np.concatenate([sol, np.zeros(max(0, n - len(sol)))])

# ============================================================
#  NLP VARIABLES
# ============================================================

cx_vars  = [ca.SX.sym(f'cx_{i}', n) for i in range(S)]
cy_vars  = [ca.SX.sym(f'cy_{i}', n) for i in range(S)]
cz_vars  = [ca.SX.sym(f'cz_{i}', n) for i in range(S)]
u_vars   = [ca.SX.sym(f'u_{i}')     for i in range(S)]
th_vars  = [ca.SX.sym(f'th_{i}')    for i in range(S)]
ph_vars  = [ca.SX.sym(f'ph_{i}')    for i in range(S)]
ud_vars  = [ca.SX.sym(f'ud_{i}')    for i in range(S)]
thd_vars = [ca.SX.sym(f'thd_{i}')   for i in range(S)]
phd_vars = [ca.SX.sym(f'phd_{i}')   for i in range(S)]
t_sym    = ca.SX.sym('t')

vars_all = ca.vertcat(*[
    v for i in range(S)
    for v in [cx_vars[i], cy_vars[i], cz_vars[i],
              u_vars[i], th_vars[i], ph_vars[i],
              ud_vars[i], thd_vars[i], phd_vars[i]]
])

lbx, ubx = [], []
for _ in range(S):
    lbx += [-ca.inf]*3*n + [1e-6, -np.pi/2, -np.pi/2, -100., -50., -50.]
    ubx += [ ca.inf]*3*n + [1e3,   np.pi/2,  np.pi/2,  100.,  50.,  50.]

# ============================================================
#  CONSTRAINTS AND COST
# ============================================================

total_cost = 0
g_cons, lbg, ubg = [], [], []

def eq(expr, val):
    g_cons.append(expr); lbg.append(float(val)); ubg.append(float(val))

def ineq_upper(expr, ub):
    g_cons.append(expr); lbg.append(-ca.inf); ubg.append(float(ub))

def ineq_lower(expr, lb):
    g_cons.append(expr); lbg.append(float(lb)); ubg.append(ca.inf)

hx, hy, hz = HOVER_START

# ── Hover start: pos + vel + acc = 0 ─────────────────────────────────────────
for c_vec, val in zip([cx_vars[0], cy_vars[0], cz_vars[0]], [hx, hy, hz]):
    eq(ca.dot(c_vec, ca.DM(time_power_vec(0.0, n))), val)
    eq(ca.dot(c_vec, ca.DM(vel_basis(0.0, n))),      0.0)
    eq(ca.dot(c_vec, ca.DM(acc_basis(0.0, n))),      0.0)

# ── Circular lap wrap: C2 ────────────────────────────────────────────────────
tf_last = float(SEGMENTS[-1]['duration'])
for c_last, c_first in zip([cx_vars[-1], cy_vars[-1], cz_vars[-1]],
                            [cx_vars[first_c], cy_vars[first_c], cz_vars[first_c]]):
    for basis in [time_power_vec, vel_basis, acc_basis]:
        eq(ca.dot(c_last,  ca.DM(basis(tf_last, n))) -
           ca.dot(c_first, ca.DM(basis(0.0,    n))), 0.0)

# ── Per-segment ───────────────────────────────────────────────────────────────
for i, seg in enumerate(SEGMENTS):
    cx = cx_vars[i]; cy = cy_vars[i]; cz = cz_vars[i]
    u_s  = u_vars[i];  th_s  = th_vars[i];  ph_s  = ph_vars[i]
    ud_s = ud_vars[i]; thd_s = thd_vars[i]; phd_s = phd_vars[i]

    tf    = float(seg['duration'])
    t_mid = tf / 2.0
    obs   = seg['obstacle']
    xm, ym, zm = obs['pos']

    phi_dot_des   = float(obs.get('phi_dot',   0.0))
    theta_dot_des = float(obs.get('theta_dot', 0.0))
    vx_des = obs.get('vx', None)
    vy_des = obs.get('vy', None)
    vz_des = obs.get('vz', None)

    inc_deg  = obs.get('inclination_deg', None)
    inc_axis = obs.get('inclination_axis', 'x')

    # Resolve attitude angles
    # With yaw=0: roll (phi) controls y-force, pitch (theta) controls x-force.
    # For axis='y' wall (motion in y-direction): phi = 90 - inc_deg, theta = 0
    # For axis='x' wall (motion in x-direction): theta = 90 - inc_deg, phi = 0
    if inc_deg is not None:
        alpha = np.deg2rad(float(inc_deg))
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        if inc_axis == 'x':
            phi_des   = np.deg2rad(float(obs.get('phi_deg',   0.0)))
            theta_des = np.deg2rad(float(obs.get('theta_deg', 90.0 - inc_deg)))
        else:  # 'y'
            phi_des   = np.deg2rad(float(obs.get('phi_deg',   90.0 - inc_deg)))
            theta_des = np.deg2rad(float(obs.get('theta_deg', 0.0)))
    else:
        phi_des   = np.deg2rad(float(obs.get('phi_deg',   0.0)))
        theta_des = np.deg2rad(float(obs.get('theta_deg', 0.0)))
        alpha = cos_a = sin_a = None

    # ── Snap cost ─────────────────────────────────────────────────────────────
    snx = sum(j*(j-1)*(j-2)*(j-3)*cx[j]*t_sym**(j-4) for j in range(4, n))
    sny = sum(j*(j-1)*(j-2)*(j-3)*cy[j]*t_sym**(j-4) for j in range(4, n))
    snz = sum(j*(j-1)*(j-2)*(j-3)*cz[j]*t_sym**(j-4) for j in range(4, n))
    dt_int = tf / float(N_INT)
    for k in range(N_INT + 1):
        ti = k * dt_int
        total_cost += ca.substitute(snx**2 + sny**2 + snz**2, t_sym, ti) * dt_int

    # ── C2 junction continuity ────────────────────────────────────────────────
    if i < S - 1:
        cx_n = cx_vars[i+1]; cy_n = cy_vars[i+1]; cz_n = cz_vars[i+1]
        for c_c, c_n in zip([cx, cy, cz], [cx_n, cy_n, cz_n]):
            for basis in [time_power_vec, vel_basis, acc_basis]:
                eq(ca.dot(c_c, ca.DM(basis(tf,  n))) -
                   ca.dot(c_n, ca.DM(basis(0.0, n))), 0.0)

    # ── Obstacle: position ────────────────────────────────────────────────────
    eq(ca.dot(cx, ca.DM(time_power_vec(t_mid, n))), xm)
    eq(ca.dot(cy, ca.DM(time_power_vec(t_mid, n))), ym)
    eq(ca.dot(cz, ca.DM(time_power_vec(t_mid, n))), zm)

    # ── Obstacle: explicit individual velocity constraints ────────────────────
    for c_vec, v_des in zip([cx, cy, cz], [vx_des, vy_des, vz_des]):
        if v_des is not None:
            eq(ca.dot(c_vec, ca.DM(vel_basis(t_mid, n))), float(v_des))

    # ── Obstacle: inclination direction constraint ────────────────────────────
    # Tangent constraint: v · t_wall = 0
    # t_wall ⊥ n in the movement plane.
    # axis='y': t_wall=(0,cos(α),sin(α)) → vy·cos(α) + vz·sin(α) = 0
    # axis='x': t_wall=(cos(α),0,sin(α)) → vx·cos(α) + vz·sin(α) = 0
    # Check α=90° (vertical): primary_vel·0 + vz·1 = 0 → vz=0 ✓
    # Check α=50°: vy·0.643 + vz·0.766 = 0 → vz = -vy·0.839
    #              with vy<0 (moving in -y): vz>0 (descending) ✓
    #
    # Approach direction: instead of v·n, we constrain the PRIMARY axis velocity
    # directly. This is unambiguous — loops cannot satisfy it.
    # approach_dir = sign(desired primary velocity):
    #   -1  →  drone moves in -y (or -x): primary_vel ≤ -MIN_PASS_SPEED
    #   +1  →  drone moves in +y (or +x): primary_vel ≥ +MIN_PASS_SPEED
    if inc_deg is not None:
        vx_s = ca.dot(cx, ca.DM(vel_basis(t_mid, n)))
        vy_s = ca.dot(cy, ca.DM(vel_basis(t_mid, n)))
        vz_s = ca.dot(cz, ca.DM(vel_basis(t_mid, n)))

        if inc_axis == 'x':
            eq(vx_s * cos_a + vz_s * sin_a, 0.0)    # tangent constraint
            # direct vx direction constraint
            approach_dir = float(obs.get('approach_dir', -1.0))
            ineq_lower(approach_dir * vx_s, MIN_PASS_SPEED)
        else:
            eq(vy_s * cos_a + vz_s * sin_a, 0.0)    # tangent constraint
            # direct vy direction constraint
            approach_dir = float(obs.get('approach_dir', -1.0))
            ineq_lower(approach_dir * vy_s, MIN_PASS_SPEED)

    # ── NED coupling at t_mid ─────────────────────────────────────────────────
    ax_m = ca.dot(cx, ca.DM(acc_basis(t_mid, n)))
    ay_m = ca.dot(cy, ca.DM(acc_basis(t_mid, n)))
    az_m = ca.dot(cz, ca.DM(acc_basis(t_mid, n)))
    vx_m = ca.dot(cx, ca.DM(vel_basis(t_mid, n)))
    vy_m = ca.dot(cy, ca.DM(vel_basis(t_mid, n)))
    vz_m = ca.dot(cz, ca.DM(vel_basis(t_mid, n)))

    fdx    = ca.DM(RHO_X)*vx_m;  fdot_x = ca.DM(RHO_X)*ax_m
    fdy    = ca.DM(RHO_Y)*vy_m;  fdot_y = ca.DM(RHO_Y)*ay_m
    fdz    = ca.DM(RHO_Z)*vz_m;  fdot_z = ca.DM(RHO_Z)*az_m

    eq((-ax_m - fdx) - u_s * ca.sin(th_s) * ca.cos(ph_s), 0.0)
    eq((-ay_m - fdy) - u_s * ca.sin(ph_s),                 0.0)
    eq((GRAV - az_m - fdz) - u_s * ca.cos(ph_s) * ca.cos(th_s), 0.0)

    eq(ph_s, phi_des)
    eq(th_s, theta_des)

    jx_c = -ca.dot(cx, ca.DM(jerk_basis(t_mid, n)))
    jy_c = -ca.dot(cy, ca.DM(jerk_basis(t_mid, n)))
    jz_c = -ca.dot(cz, ca.DM(jerk_basis(t_mid, n)))

    bx_dot = jx_c - fdot_x
    by_dot = jy_c - fdot_y
    bz_dot = jz_c - fdot_z

    s_th = ca.sin(th_s); c_th = ca.cos(th_s)
    s_ph = ca.sin(ph_s); c_ph = ca.cos(ph_s)

    A11 =  s_th*c_ph;  A12 =  u_s*c_th*c_ph;  A13 = -u_s*s_th*s_ph
    A21 = -s_ph;        A22 =  0.0;              A23 = -u_s*c_ph
    A31 =  c_th*c_ph;  A32 = -u_s*c_ph*s_th;   A33 = -u_s*s_ph*c_th

    eq(A11*ud_s + A12*thd_s + A13*phd_s - bx_dot, 0.0)
    eq(A21*ud_s + A22*thd_s + A23*phd_s - by_dot, 0.0)
    eq(A31*ud_s + A32*thd_s + A33*phd_s - bz_dot, 0.0)
    eq(phd_s, phi_dot_des)
    eq(thd_s, theta_dot_des)

    # ── No negative thrust ────────────────────────────────────────────────────
    for k in range(M_SAMP):
        ti = (k / float(max(1, M_SAMP - 1))) * tf
        ineq_upper(ca.dot(cz, ca.DM(acc_basis(ti, n))), GRAV - MARGIN)

    # ── Hard workspace box ────────────────────────────────────────────────────
    for k in range(B_SAMP):
        ti = (k / float(max(1, B_SAMP - 1))) * tf
        xi = ca.dot(cx, ca.DM(time_power_vec(ti, n)))
        yi = ca.dot(cy, ca.DM(time_power_vec(ti, n)))
        zi = ca.dot(cz, ca.DM(time_power_vec(ti, n)))
        ineq_upper( xi,  X_MAX)
        ineq_upper(-xi, -X_MIN)
        ineq_upper( yi,  Y_MAX)
        ineq_upper(-yi, -Y_MIN)
        ineq_upper( zi,  Z_MAX)

# ============================================================
#  INITIAL GUESS  (chained: seg i starts near obstacle i-1)
# ============================================================

x0_list = []
for i, seg in enumerate(SEGMENTS):
    tf = float(seg['duration']); t_mid = tf/2.0
    obs = seg['obstacle']; xm, ym, zm = obs['pos']

    xs, ys, zs = (hx, hy, hz) if i == 0 else SEGMENTS[i-1]['obstacle']['pos']
    xe, ye, ze = SEGMENTS[i+1]['obstacle']['pos'] if i < S-1 else SEGMENTS[first_c]['obstacle']['pos']

    cx0 = fit_initial_coeffs([('pos', 0.0, xs), ('pos', t_mid, xm), ('pos', tf, xe)], n)
    cy0 = fit_initial_coeffs([('pos', 0.0, ys), ('pos', t_mid, ym), ('pos', tf, ye)], n)
    cz0 = fit_initial_coeffs([('pos', 0.0, zs), ('pos', t_mid, zm), ('pos', tf, ze)], n)

    inc_deg  = obs.get('inclination_deg')
    inc_axis = obs.get('inclination_axis', 'x')
    if inc_deg is not None:
        theta_g = float(obs.get('theta_deg', inc_deg-90. if inc_axis=='x' else 0.))
        phi_g   = float(obs.get('phi_deg',   0.          if inc_axis=='x' else 90.-inc_deg))
    else:
        theta_g = float(obs.get('theta_deg', 0.))
        phi_g   = float(obs.get('phi_deg',   0.))

    x0_list.append(np.concatenate([cx0, cy0, cz0,
        [GRAV, np.deg2rad(theta_g), np.deg2rad(phi_g),
         0., float(obs.get('theta_dot', 0.)), float(obs.get('phi_dot', 0.))]]))

x0_full = np.concatenate(x0_list)

# ============================================================
#  SOLVE
# ============================================================

print(f"Constraints: {len(g_cons)}\n")

nlp    = {'x': vars_all, 'f': total_cost, 'g': ca.vertcat(*g_cons)}
opts   = {'ipopt.print_level': 3, 'print_time': True,
          'ipopt.max_iter': 3000, 'ipopt.tol': 1e-6}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
sol    = solver(x0=x0_full.tolist(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
solx   = sol['x'].full().flatten()

print(f"\nSolver status: {solver.stats()['return_status']}")

# ============================================================
#  EXTRACT + DIAGNOSTICS
# ============================================================

n_per  = 3*n + 6
solved = []

print("\n── Per-segment results ──────────────────────────────────────────────")
for i in range(S):
    blk    = solx[i*n_per:(i+1)*n_per]
    cx_sol = blk[0:n]; cy_sol = blk[n:2*n]; cz_sol = blk[2*n:3*n]
    u_s, th_s, ph_s = blk[3*n], blk[3*n+1], blk[3*n+2]
    tf  = float(SEGMENTS[i]['duration'])
    t_mid = tf / 2.0
    obs = SEGMENTS[i]['obstacle']

    label = 'ENTRY' if i < N_ENTRY_SEGMENTS else f'CIRC-{i-N_ENTRY_SEGMENTS+1}'
    inc   = obs.get('inclination_deg')
    inc_str = f' [inc={inc:.0f}° axis={obs.get("inclination_axis","x")}]' if inc else ''

    pos_s = np.round([eval_poly(cx_sol,0.),  eval_poly(cy_sol,0.),  eval_poly(cz_sol,0.)],  3)
    pos_e = np.round([eval_poly(cx_sol,tf),  eval_poly(cy_sol,tf),  eval_poly(cz_sol,tf)],  3)
    vel_m = [eval_vel(cx_sol,t_mid), eval_vel(cy_sol,t_mid), eval_vel(cz_sol,t_mid)]
    v_mag = np.sqrt(sum(v**2 for v in vel_m))

    print(f"  seg{i}[{label}]{inc_str}:")
    print(f"    θ={np.rad2deg(th_s):.1f}°  φ={np.rad2deg(ph_s):.1f}°  u={u_s:.3f}")
    print(f"    pos: {tuple(pos_s)} → {tuple(pos_e)}")
    print(f"    vel@t_mid: ({vel_m[0]:.3f}, {vel_m[1]:.3f}, {vel_m[2]:.3f})  |v|={v_mag:.3f}")

    if inc is not None:
        alpha    = np.deg2rad(float(inc))
        inc_axis = obs.get('inclination_axis', 'x')
        app_dir  = float(obs.get('approach_dir', -1.0))
        if inc_axis == 'x':
            tang_r   = vel_m[0]*np.cos(alpha) + vel_m[2]*np.sin(alpha)
            prim_vel = vel_m[0]
        else:
            tang_r   = vel_m[1]*np.cos(alpha) + vel_m[2]*np.sin(alpha)
            prim_vel = vel_m[1]
        print(f"    tang_res={tang_r:.4f}  primary_vel={prim_vel:.3f}  "
              f"(approach_dir={app_dir:+.0f}, need {app_dir*prim_vel:.3f} ≥ {MIN_PASS_SPEED})")

    # Box check
    ts_c = np.linspace(0, tf, 200)
    xs_c = np.array([eval_poly(cx_sol,t) for t in ts_c])
    ys_c = np.array([eval_poly(cy_sol,t) for t in ts_c])
    zs_c = np.array([eval_poly(cz_sol,t) for t in ts_c])
    vx_b = max(0, np.max(xs_c)-X_MAX, X_MIN-np.min(xs_c))
    vy_b = max(0, np.max(ys_c)-Y_MAX, Y_MIN-np.min(ys_c))
    vz_b = max(0, np.max(zs_c)-Z_MAX)
    bv   = max(vx_b, vy_b, vz_b)
    print(f"    box: Δx={vx_b:.3f}  Δy={vy_b:.3f}  Δz={vz_b:.3f}"
          + (" ← VIOLATION" if bv > 0.01 else " ✓"))

    solved.append({'cx': cx_sol.tolist(), 'cy': cy_sol.tolist(), 'cz': cz_sol.tolist(),
                   'tf': tf, 't_mid': t_mid})

print("\n── Lap-wrap C2 continuity check ─────────────────────────────────────")
last = solved[-1]; fcs = solved[first_c]
for axis, cl, cf in zip(['x','y','z'],
                         [last['cx'], last['cy'], last['cz']],
                         [fcs['cx'],  fcs['cy'],  fcs['cz']]):
    dp = eval_poly(cl,last['tf'])-eval_poly(cf,0.)
    dv = eval_vel(cl, last['tf'])-eval_vel(cf, 0.)
    da = eval_acc(cl, last['tf'])-eval_acc(cf, 0.)
    print(f"  {axis}: Δpos={dp:.2e}  Δvel={dv:.2e}  Δacc={da:.2e}")

# ============================================================
#  YAML OUTPUTS
# ============================================================

ros2 = {'/**': {'ros__parameters': {'autopilot': {'polynomial_trajectory': {
    'num_laps'             : N_LAPS,
    'num_entry_segments'   : N_ENTRY_SEGMENTS,
    'num_circular_segments': N_CIRC,
    'num_segments_total'   : S,
    'entry_duration'       : round(entry_dur, 6),
    'circular_lap_duration': round(circ_dur,  6),
    'drag': {'rho_x': RHO_X, 'rho_y': RHO_Y, 'rho_z': RHO_Z},
    'segments': {},
}}}}}

sd = ros2['/**']['ros__parameters']['autopilot']['polynomial_trajectory']['segments']
for i, res in enumerate(solved):
    obs = SEGMENTS[i]['obstacle']
    inc = float(obs.get('inclination_deg', 90.0))
    iax = str(obs.get('inclination_axis', 'x'))
    if obs.get('inclination_deg') is not None:
        if iax == 'x':
            th_o = float(obs.get('theta_deg', 90.-inc))
            ph_o = float(obs.get('phi_deg',   0.))
        else:  # 'y'
            th_o = float(obs.get('theta_deg', 0.))
            ph_o = float(obs.get('phi_deg',   90.-inc))
    else:
        th_o = float(obs.get('theta_deg', 0.))
        ph_o = float(obs.get('phi_deg',   0.))
    sd[f'seg{i}'] = {
        'duration': round(res['tf'], 6),
        'x': [round(v,10) for v in res['cx']],
        'y': [round(v,10) for v in res['cy']],
        'z': [round(v,10) for v in res['cz']],
        'obstacle': {
            'pos'             : [round(v,6) for v in obs['pos']],
            'phi_deg'         : round(ph_o, 4),
            'theta_deg'       : round(th_o, 4),
            'inclination_deg' : round(inc,  4),
            'inclination_axis': iax,
        },
    }

with open('polynomial_trajectory.yaml', 'w') as f:
    yaml.dump(ros2, f, default_flow_style=False, sort_keys=False)
print("\nSaved 'polynomial_trajectory.yaml'")

# ============================================================
#  PLOTTING
# ============================================================

WINDOW_SIZE = 0.4

def draw_window_3d(ax, obs, color, idx):
    xm, ym, zm = obs['pos']
    center = np.array([xm, ym, zm])
    inc   = float(obs.get('inclination_deg', 90.0))
    iax   = obs.get('inclination_axis', 'x')
    ph    = np.deg2rad(float(obs.get('phi_deg', 0.0) if obs.get('inclination_deg') is None
                             else obs.get('phi_deg', 90.0 - inc) if iax=='y'
                             else obs.get('phi_deg', 0.0)))
    alpha = np.deg2rad(inc)

    if iax == 'x':
        # n = (-sin(α), 0, +cos(α)) in NED
        normal = np.array([-np.sin(alpha), 0., +np.cos(alpha)])
        e1 = np.array([0., 1., 0.])
        e2 = np.array([np.cos(alpha), 0., np.sin(alpha)])
    else:
        # n = (0, -sin(α), +cos(α)) in NED
        normal = np.array([0., -np.sin(alpha), +np.cos(alpha)])
        e1 = np.array([1., 0., 0.])
        e2 = np.array([0., np.cos(alpha), np.sin(alpha)])
    normal /= np.linalg.norm(normal)
    e1r = np.cos(ph)*e1 + np.sin(ph)*e2
    e2r = -np.sin(ph)*e1 + np.cos(ph)*e2

    hw = WINDOW_SIZE / 2.0
    corners = np.array([center+hw*e1r+hw*e2r, center-hw*e1r+hw*e2r,
                         center-hw*e1r-hw*e2r, center+hw*e1r-hw*e2r])
    ax.add_collection3d(Poly3DCollection([corners], alpha=0.35,
                                          facecolor=color, edgecolor=color, linewidth=1.8))
    al = 0.35
    ax.quiver(xm, ym, zm, normal[0]*al, normal[1]*al, normal[2]*al,
              color='gold', linewidth=2.0, arrow_length_ratio=0.35)
    ax.text(xm+normal[0]*0.08, ym+normal[1]*0.08, zm+normal[2]*0.08,
            f' W{idx}', fontsize=8, color=color)

ec = 'gray'
cc = plt.cm.tab10(np.linspace(0, 1, N_CIRC))

fig3d = plt.figure(figsize=(10, 7))
ax3d  = fig3d.add_subplot(111, projection='3d')
t_off = 0.0

for i, res in enumerate(solved):
    ts = np.linspace(0., res['tf'], 300)
    cx, cy, cz = res['cx'], res['cy'], res['cz']
    xs = np.array([eval_poly(cx,t) for t in ts])
    ys = np.array([eval_poly(cy,t) for t in ts])
    zs = np.array([eval_poly(cz,t) for t in ts])
    color = ec if i < N_ENTRY_SEGMENTS else cc[i-N_ENTRY_SEGMENTS]
    lbl   = 'Entry' if i < N_ENTRY_SEGMENTS else f'C{i-N_ENTRY_SEGMENTS+1}'
    ax3d.plot(xs, ys, zs, color=color, linewidth=2., label=lbl)
    draw_window_3d(ax3d, SEGMENTS[i]['obstacle'], color, i)
    t_off += res['tf']

ax3d.scatter(*HOVER_START, color='black', s=120, marker='*', label='Start')
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.set_title(f'C2+snap  ORDER={ORDER}  |  inclined W2: {SEGMENTS[2]["obstacle"]["inclination_deg"]}°')
ax3d.legend(fontsize=8)
plt.tight_layout()
plt.show(block=False)

# Time series
all_ts=[]; all_p={'x':[],'y':[],'z':[]}; all_v={'x':[],'y':[],'z':[]}; all_a={'x':[],'y':[],'z':[]}
t_off = 0.
for res in solved:
    ts = np.linspace(0., res['tf'], 300)
    cx, cy, cz = res['cx'], res['cy'], res['cz']
    all_ts.append(ts+t_off)
    all_p['x'].append(np.array([eval_poly(cx,t) for t in ts]))
    all_p['y'].append(np.array([eval_poly(cy,t) for t in ts]))
    all_p['z'].append(np.array([eval_poly(cz,t) for t in ts]))
    all_v['x'].append(np.array([eval_vel(cx,t) for t in ts]))
    all_v['y'].append(np.array([eval_vel(cy,t) for t in ts]))
    all_v['z'].append(np.array([eval_vel(cz,t) for t in ts]))
    all_a['x'].append(np.array([eval_acc(cx,t) for t in ts]))
    all_a['y'].append(np.array([eval_acc(cy,t) for t in ts]))
    all_a['z'].append(np.array([eval_acc(cz,t) for t in ts]))
    t_off += res['tf']

ts_c = np.concatenate(all_ts)
fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for k, l, c in zip(['x','y','z'], ['x','y','z'], ['tab:blue','tab:orange','tab:green']):
    axs[0].plot(ts_c, np.concatenate(all_p[k]), color=c, label=l)
    axs[1].plot(ts_c, np.concatenate(all_v[k]), color=c, label=l)
    axs[2].plot(ts_c, np.concatenate(all_a[k]), color=c, label=l)
axs[2].axhline(GRAV-MARGIN, color='r', linestyle='--', linewidth=0.8, label='a_z limit')
t_b = 0.
for res in solved:
    for ax in axs: ax.axvline(t_b, color='gray', linestyle=':', linewidth=0.8)
    t_b += res['tf']
axs[0].set_ylabel('Pos [m]');        axs[0].legend(); axs[0].grid(True)
axs[1].set_ylabel('Vel [m/s]');      axs[1].legend(); axs[1].grid(True)
axs[2].set_ylabel('Acc [m/s²]');     axs[2].set_xlabel('Time [s]')
axs[2].legend(); axs[2].grid(True)
fig.suptitle(f'C2 + snap  ORDER={ORDER}')
plt.tight_layout()
plt.show()