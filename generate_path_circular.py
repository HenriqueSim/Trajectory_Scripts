#!/usr/bin/env python3
"""
Multi-segment circular trajectory generator — global NLP  (v6)

Structure
---------
  Segment 0          : entry  — played exactly ONCE.
                       Starts at HOVER_START (free vel/acc — controller converges).
                       Ends with C2 continuity into the first circular segment.

  Segments 1 .. N_CIRC : circular lap — repeated N_LAPS times.
                       C2 continuity at every internal junction AND at the
                       lap-wrap boundary (end of last circ → start of first circ).

Key parameters
--------------
  HOVER_START        : drone position at t=0 (where it is hovering before entry)
  N_ENTRY_SEGMENTS   : number of entry segments (usually 1)
  N_LAPS             : number of times the circular portion repeats
  SEGMENTS           : list of dicts — entry segment(s) first, then circular
  ORDER              : polynomial degree per segment (11 is fine for 1 obstacle/seg)

YAML outputs
------------
  circular_trajectory.yaml   — human-readable debug
  polynomial_trajectory.yaml — ROS 2 parameter format, copy to autopilot/config/
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import yaml
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
#  USER PARAMETERS
# ============================================================

N_LAPS           = 2    # circular laps to repeat
N_ENTRY_SEGMENTS = 1    # entry segments (played once, not repeated)
ORDER            = 13   # polynomial degree per segment.
                        # C4 junction continuity uses 5 constraints/axis/junction.
                        # Middle circular segments: 5(start)+5(end)+2(obs)=12 constraints/axis.
                        # ORDER=13 → 14 coefficients/axis → 2 DOF/axis for snap minimisation.
N_INT            = 80   # snap-cost integration samples per segment
M_SAMP           = 60   # a_z inequality samples per segment
B_SAMP           = 40   # box constraint samples per segment (fewer is fine, these are smooth)
MARGIN           = 1e-3
GRAV             = 9.81

# ── Workspace box constraints (sampled along each segment) ────────────────────
# NED frame: x = North, y = East, z = Down (negative = up)
# z < Z_CEIL  enforces a minimum altitude (e.g. -0.5 → at least 0.5 m above ground)
X_MIN, X_MAX = -3.0,  3.0
Y_MIN, Y_MAX = -2.0,  2.0
Z_CEIL       = -0.5   # drone must stay at z < -0.5 (above 0.5 m floor)

RHO_X, RHO_Y, RHO_Z = 0.0, 0.0, 0.0

# Position the drone is hovering at when the mode starts.
# The entry segment begins here (vel=0, acc=0 fixed at t=0).
HOVER_START = (2.0, -1.0, -1.5)

# ── Segments ────────────────────────────────────────────────────────────────
# First N_ENTRY_SEGMENTS entries are the entry (played once).
# The rest are the circular lap (repeated N_LAPS times).
#
# Obstacle passage is always at t_mid = duration/2 of each segment.
# vx/vy/vz = None  →  that velocity axis is unconstrained at the obstacle.

SEGMENTS = [
    # ── Entry segment ────────────────────────────────────────────────────────
    {
        'duration' : 2.5,
        'obstacle' : {
            'pos'       : (2.0, 0.0, -1.5),
            'phi_deg'   : 0.0,
            'theta_deg' : 20.0,
            'phi_dot'   : 0.0,
            'theta_dot' : 0.0,
            'vx'        : 0.0,
            'vy'        : None,
            'vz'        : 0.0,
        },
    },
    # ── Circular segment 1 ───────────────────────────────────────────────────
    {
        'duration' : 3.0,
        'obstacle' : {
            'pos'       : (0.0, 1.0, -1.5),
            'phi_deg'   : 20.0,
            'theta_deg' : 0.0,
            'phi_dot'   : 0.0,
            'theta_dot' : 0.0,
            'vx'        : None,
            'vy'        : 0.0,
            'vz'        : 0.0,
        },
    },
    # ── Circular segment 2 ───────────────────────────────────────────────────
    {
        'duration' : 2.5,
        'obstacle' : {
            'pos'       : (-2.0, 0.0, -1.5),
            'phi_deg'   : 0.0,
            'theta_deg' : 20.0,
            'phi_dot'   : 0.0,
            'theta_dot' : 0.0,
            'vx'        : 0.0,
            'vy'        : None,
            'vz'        : 0.0,
        },
    },
    # ── Circular segment 3 ───────────────────────────────────────────────────
    {
        'duration' : 3.0,
        'obstacle' : {
            'pos'       : (0.0, -1.0, -1.5),
            'phi_deg'   : 20.0,
            'theta_deg' : 0.0,
            'phi_dot'   : 0.0,
            'theta_dot' : 0.0,
            'vx'        : None,
            'vy'        : 0.0,
            'vz'        : 0.0,
        },
    },
    # ── Circular segment 4 ───────────────────────────────────────────────────
    {
        'duration' : 2.5,
        'obstacle' : {
            'pos'       : (2.0, 0.0, -1.5),
            'phi_deg'   : 0.0,
            'theta_deg' : 20.0,
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

S      = len(SEGMENTS)
N_CIRC = S - N_ENTRY_SEGMENTS          # number of circular segments
assert N_CIRC >= 1, "Need at least one circular segment"

entry_dur   = sum(SEGMENTS[i]['duration'] for i in range(N_ENTRY_SEGMENTS))
circ_dur    = sum(SEGMENTS[i]['duration'] for i in range(N_ENTRY_SEGMENTS, S))
total_dur   = entry_dur + N_LAPS * circ_dur

n = ORDER + 1  # coefficients per axis per segment

print(f"\nSegments: {N_ENTRY_SEGMENTS} entry + {N_CIRC} circular")
print(f"Entry duration: {entry_dur:.2f}s  |  Circular lap: {circ_dur:.2f}s")
print(f"Total ({N_LAPS} laps): {total_dur:.2f}s  |  ORDER={ORDER}  |  "
      f"Decision vars: {S*(3*n+6)}")

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

def snap_basis(t, n):
    return np.array([0.0 if i <= 3 else i*(i-1)*(i-2)*(i-3)*t**(i-4) for i in range(n)], dtype=float)

def eval_poly(c, t):
    return sum(c[i]*t**i for i in range(len(c)))

def eval_vel(c, t):
    return sum(i*c[i]*t**(i-1) for i in range(1, len(c)))

def eval_acc(c, t):
    return sum(i*(i-1)*c[i]*t**(i-2) for i in range(2, len(c)))

def eval_jerk(c, t):
    return sum(i*(i-1)*(i-2)*c[i]*t**(i-3) for i in range(3, len(c)))

def eval_snap(c, t):
    return sum(i*(i-1)*(i-2)*(i-3)*c[i]*t**(i-4) for i in range(4, len(c)))

def fit_initial_coeffs(constraints, n):
    bases = {'pos': time_power_vec, 'vel': vel_basis, 'acc': acc_basis}
    A, b = [], []
    for typ, tval, val in constraints:
        A.append(bases[typ](tval, n))
        b.append(val)
    if not A:
        return np.zeros(n)
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

t_sym = ca.SX.sym('t')

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

def ineq_range(expr, lb, ub):
    """lb <= expr <= ub"""
    g_cons.append(expr); lbg.append(float(lb)); ubg.append(float(ub))

hx, hy, hz = HOVER_START

# ── Hover start: fix position, velocity, acceleration and jerk at t=0 ─────────
# Snap is left free — fixing 4 derivatives per axis uses 12 of 14 coefficients,
# leaving 2 DOF per axis for the snap-minimising optimiser.
for c_vec, val in zip([cx_vars[0], cy_vars[0], cz_vars[0]], [hx, hy, hz]):
    eq(ca.dot(c_vec, ca.DM(time_power_vec(0.0, n))), val)  # position
    eq(ca.dot(c_vec, ca.DM(vel_basis(0.0,        n))), 0.0)  # velocity  = 0
    eq(ca.dot(c_vec, ca.DM(acc_basis(0.0,        n))), 0.0)  # acceleration = 0
    eq(ca.dot(c_vec, ca.DM(jerk_basis(0.0,       n))), 0.0)  # jerk = 0

# ── Circular lap wrap: C4 continuity (pos + vel + acc + jerk + snap) ─────────
# End of last circular segment must match start of first circular segment.
tf_last   = float(SEGMENTS[-1]['duration'])
first_c   = N_ENTRY_SEGMENTS
for c_last, c_first in zip([cx_vars[-1], cy_vars[-1], cz_vars[-1]],
                             [cx_vars[first_c], cy_vars[first_c], cz_vars[first_c]]):
    for basis in [time_power_vec, vel_basis, acc_basis, jerk_basis, snap_basis]:
        eq(ca.dot(c_last,  ca.DM(basis(tf_last, n))) -
           ca.dot(c_first, ca.DM(basis(0.0,    n))), 0.0)

# ── Per-segment constraints ───────────────────────────────────────────────────
for i, seg in enumerate(SEGMENTS):
    cx = cx_vars[i]; cy = cy_vars[i]; cz = cz_vars[i]
    u_s  = u_vars[i];  th_s  = th_vars[i];  ph_s  = ph_vars[i]
    ud_s = ud_vars[i]; thd_s = thd_vars[i]; phd_s = phd_vars[i]

    tf    = float(seg['duration'])
    t_mid = tf / 2.0
    obs   = seg['obstacle']

    xm, ym, zm    = obs['pos']
    phi_des       = np.deg2rad(obs['phi_deg'])
    theta_des     = np.deg2rad(obs['theta_deg'])
    phi_dot_des   = float(obs.get('phi_dot',   0.0))
    theta_dot_des = float(obs.get('theta_dot', 0.0))
    vx_des = obs.get('vx', None)
    vy_des = obs.get('vy', None)
    vz_des = obs.get('vz', None)

    # Snap cost
    snap_x = sum(j*(j-1)*(j-2)*(j-3)*cx[j]*t_sym**(j-4) for j in range(4, n))
    snap_y = sum(j*(j-1)*(j-2)*(j-3)*cy[j]*t_sym**(j-4) for j in range(4, n))
    snap_z = sum(j*(j-1)*(j-2)*(j-3)*cz[j]*t_sym**(j-4) for j in range(4, n))
    dt_int = tf / float(N_INT)
    for k in range(N_INT + 1):
        ti = k * dt_int
        total_cost += ca.substitute(snap_x**2 + snap_y**2 + snap_z**2, t_sym, ti) * dt_int

    # C4 continuity to next segment (pos, vel, acc, jerk, snap)
    if i < S - 1:
        cx_n = cx_vars[i+1]; cy_n = cy_vars[i+1]; cz_n = cz_vars[i+1]
        for c_cur, c_nxt in zip([cx, cy, cz], [cx_n, cy_n, cz_n]):
            for basis in [time_power_vec, vel_basis, acc_basis, jerk_basis, snap_basis]:
                eq(ca.dot(c_cur, ca.DM(basis(tf,  n))) -
                   ca.dot(c_nxt, ca.DM(basis(0.0, n))), 0.0)

    # Obstacle: position at t_mid
    eq(ca.dot(cx, ca.DM(time_power_vec(t_mid, n))), xm)
    eq(ca.dot(cy, ca.DM(time_power_vec(t_mid, n))), ym)
    eq(ca.dot(cz, ca.DM(time_power_vec(t_mid, n))), zm)

    # Obstacle: optional velocity constraints
    for c_vec, v_des in zip([cx, cy, cz], [vx_des, vy_des, vz_des]):
        if v_des is not None:
            eq(ca.dot(c_vec, ca.DM(vel_basis(t_mid, n))), float(v_des))

    # Symbolic acc and vel at t_mid
    ax_m = ca.dot(cx, ca.DM(acc_basis(t_mid, n)))
    ay_m = ca.dot(cy, ca.DM(acc_basis(t_mid, n)))
    az_m = ca.dot(cz, ca.DM(acc_basis(t_mid, n)))
    vx_m = ca.dot(cx, ca.DM(vel_basis(t_mid, n)))
    vy_m = ca.dot(cy, ca.DM(vel_basis(t_mid, n)))
    vz_m = ca.dot(cz, ca.DM(vel_basis(t_mid, n)))

    # Linear drag
    fdx    = ca.DM(RHO_X)*vx_m;  fdot_x = ca.DM(RHO_X)*ax_m
    fdy    = ca.DM(RHO_Y)*vy_m;  fdot_y = ca.DM(RHO_Y)*ay_m
    fdz    = ca.DM(RHO_Z)*vz_m;  fdot_z = ca.DM(RHO_Z)*az_m

    # NED coupling
    eq((-ax_m - fdx) - u_s * ca.sin(th_s) * ca.cos(ph_s), 0.0)
    eq((-ay_m - fdy) - u_s * ca.sin(ph_s),                 0.0)
    eq((GRAV - az_m - fdz) - u_s * ca.cos(ph_s) * ca.cos(th_s), 0.0)

    # Desired roll and pitch
    eq(ph_s, phi_des)
    eq(th_s, theta_des)

    # Jerk → Euler-rate
    jx_c   = -ca.dot(cx, ca.DM(jerk_basis(t_mid, n)))
    jy_c   = -ca.dot(cy, ca.DM(jerk_basis(t_mid, n)))
    jz_c   = -ca.dot(cz, ca.DM(jerk_basis(t_mid, n)))
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

    eq(phd_s,  phi_dot_des)
    eq(thd_s, theta_dot_des)

    # No-negative-thrust
    for k in range(M_SAMP):
        ti = (k / float(max(1, M_SAMP - 1))) * tf
        ineq_upper(ca.dot(cz, ca.DM(acc_basis(ti, n))), GRAV - MARGIN)

    # ── Workspace box constraints ─────────────────────────────────────────────
    # Sampled at B_SAMP points including endpoints.
    # x ∈ [X_MIN, X_MAX],  y ∈ [Y_MIN, Y_MAX],  z < Z_CEIL (NED, so z is negative)
    for k in range(B_SAMP):
        ti = (k / float(max(1, B_SAMP - 1))) * tf
        xi = ca.dot(cx, ca.DM(time_power_vec(ti, n)))
        yi = ca.dot(cy, ca.DM(time_power_vec(ti, n)))
        zi = ca.dot(cz, ca.DM(time_power_vec(ti, n)))
        ineq_range(xi, X_MIN, X_MAX)
        ineq_range(yi, Y_MIN, Y_MAX)
        ineq_upper(zi, Z_CEIL)

# ============================================================
#  INITIAL GUESS
# ============================================================

x0_list = []
for i, seg in enumerate(SEGMENTS):
    tf    = float(seg['duration'])
    t_mid = tf / 2.0
    obs   = seg['obstacle']
    xm, ym, zm = obs['pos']

    # Start guess: hover_start for entry seg; previous seg end for circular
    xs = hx; ys = hy; zs = hz   # fallback
    xe = xm; ye = ym; ze = zm   # fallback end

    cx0 = fit_initial_coeffs([('pos', 0.0, xs), ('pos', t_mid, xm), ('pos', tf, xe)], n)
    cy0 = fit_initial_coeffs([('pos', 0.0, ys), ('pos', t_mid, ym), ('pos', tf, ye)], n)
    cz0 = fit_initial_coeffs([('pos', 0.0, zs), ('pos', t_mid, zm), ('pos', tf, ze)], n)

    x0_list.append(np.concatenate([
        cx0, cy0, cz0,
        [GRAV,
         np.deg2rad(obs['theta_deg']),
         np.deg2rad(obs['phi_deg']),
         0.0,
         float(obs.get('theta_dot', 0.0)),
         float(obs.get('phi_dot',   0.0))]
    ]))

x0_full = np.concatenate(x0_list)

# ============================================================
#  SOLVE
# ============================================================

nlp  = {'x': vars_all, 'f': total_cost, 'g': ca.vertcat(*g_cons)}
opts = {
    'ipopt.print_level': 3,
    'print_time'       : True,
    'ipopt.max_iter'   : 3000,
    'ipopt.tol'        : 1e-6,
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

print(f"Constraints: {len(g_cons)}\n")

sol  = solver(x0=x0_full.tolist(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
solx = sol['x'].full().flatten()

print(f"\nSolver status: {solver.stats()['return_status']}")

# ============================================================
#  EXTRACT SOLUTION
# ============================================================

n_per  = 3*n + 6
solved = []

for i in range(S):
    blk    = solx[i*n_per : (i+1)*n_per]
    cx_sol = blk[0:n]; cy_sol = blk[n:2*n]; cz_sol = blk[2*n:3*n]
    u_s, th_s, ph_s = blk[3*n], blk[3*n+1], blk[3*n+2]

    tf    = float(SEGMENTS[i]['duration'])
    label = 'ENTRY' if i < N_ENTRY_SEGMENTS else f'CIRC-{i - N_ENTRY_SEGMENTS + 1}'
    pos_s = np.round([eval_poly(cx_sol, 0.),  eval_poly(cy_sol, 0.),  eval_poly(cz_sol, 0.)],  3)
    pos_e = np.round([eval_poly(cx_sol, tf),  eval_poly(cy_sol, tf),  eval_poly(cz_sol, tf)],  3)
    vel_e = np.round([eval_vel(cx_sol,  tf),  eval_vel(cy_sol,  tf),  eval_vel(cz_sol,  tf)],  3)
    acc_e = np.round([eval_acc(cx_sol,  tf),  eval_acc(cy_sol,  tf),  eval_acc(cz_sol,  tf)],  3)

    print(f"  [{label}] seg{i}: u={u_s:.3f}  θ={np.rad2deg(th_s):.2f}°  φ={np.rad2deg(ph_s):.2f}°")
    print(f"    pos: {tuple(pos_s)} → {tuple(pos_e)}")
    if i < S - 1:
        print(f"    junction vel={tuple(vel_e)}  acc={tuple(acc_e)}")

    solved.append({'cx': cx_sol.tolist(), 'cy': cy_sol.tolist(), 'cz': cz_sol.tolist(),
                   'tf': tf, 't_mid': tf/2.0})

# Verify circular lap wrap C4 continuity
print("\n── Lap-wrap continuity check (C4) ──────────────────────────────────")
last = solved[-1]; first_c_sol = solved[N_ENTRY_SEGMENTS]
tf_l = last['tf']
for axis, cl, cf in zip(['x','y','z'],
                         [last['cx'],        last['cy'],        last['cz']],
                         [first_c_sol['cx'], first_c_sol['cy'], first_c_sol['cz']]):
    dp  = eval_poly(cl, tf_l) - eval_poly(cf, 0.)
    dv  = eval_vel(cl,  tf_l) - eval_vel(cf,  0.)
    da  = eval_acc(cl,  tf_l) - eval_acc(cf,  0.)
    dj  = eval_jerk(cl, tf_l) - eval_jerk(cf, 0.)
    ds  = eval_snap(cl, tf_l) - eval_snap(cf, 0.)
    print(f"  {axis}: Δpos={dp:.2e}  Δvel={dv:.2e}  Δacc={da:.2e}  "
          f"Δjerk={dj:.2e}  Δsnap={ds:.2e}")

# ============================================================
#  YAML OUTPUTS
# ============================================================

# ── Debug YAML ───────────────────────────────────────────────────────────────
yaml_debug = {
    'num_laps'            : N_LAPS,
    'num_entry_segments'  : N_ENTRY_SEGMENTS,
    'num_circular_segments': N_CIRC,
    'entry_duration'      : round(entry_dur, 6),
    'circular_lap_duration': round(circ_dur, 6),
    'total_duration'      : round(total_dur, 6),
    'polynomial_order'    : ORDER,
    'gravity'             : GRAV,
    'drag_rho'            : {'rho_x': RHO_X, 'rho_y': RHO_Y, 'rho_z': RHO_Z},
    'hover_start'         : list(HOVER_START),
    'segments'            : [],
}

# Entry (played once)
for i in range(N_ENTRY_SEGMENTS):
    res = solved[i]
    yaml_debug['segments'].append({
        'type'        : 'entry',
        'segment'     : i,
        't_start'     : round(sum(solved[j]['tf'] for j in range(i)), 6),
        'duration'    : round(res['tf'], 6),
        'obstacle'    : {'pos': list(SEGMENTS[i]['obstacle']['pos'])},
        'coefficients': {'x': res['cx'], 'y': res['cy'], 'z': res['cz']},
    })

# Circular — listed per lap for timing reference
global_t = entry_dur
for lap in range(N_LAPS):
    for i in range(N_ENTRY_SEGMENTS, S):
        res = solved[i]
        yaml_debug['segments'].append({
            'type'        : 'circular',
            'lap'         : lap + 1,
            'segment'     : i - N_ENTRY_SEGMENTS + 1,
            't_start'     : round(global_t, 6),
            'duration'    : round(res['tf'], 6),
            'obstacle'    : {'pos': list(SEGMENTS[i]['obstacle']['pos'])},
            'coefficients': {'x': res['cx'], 'y': res['cy'], 'z': res['cz']},
        })
        global_t += res['tf']

with open('circular_trajectory.yaml', 'w') as f:
    yaml.dump(yaml_debug, f, default_flow_style=False, sort_keys=False)

print(f"\nSaved 'circular_trajectory.yaml'")

# ── ROS 2 parameter YAML ─────────────────────────────────────────────────────
ros2_params = {
    '/**': {
        'ros__parameters': {
            'autopilot': {
                'polynomial_trajectory': {
                    'num_laps'             : N_LAPS,
                    'num_entry_segments'   : N_ENTRY_SEGMENTS,
                    'num_circular_segments': N_CIRC,
                    'num_segments_total'   : S,
                    'entry_duration'       : round(entry_dur, 6),
                    'circular_lap_duration': round(circ_dur, 6),
                    'drag'                 : {'rho_x': RHO_X, 'rho_y': RHO_Y, 'rho_z': RHO_Z},
                    'segments'             : {},
                }
            }
        }
    }
}

seg_dict = ros2_params['/**']['ros__parameters']['autopilot']['polynomial_trajectory']['segments']
for i, res in enumerate(solved):
    seg_dict[f'seg{i}'] = {
        'duration': round(res['tf'], 6),
        'x'       : [round(v, 10) for v in res['cx']],
        'y'       : [round(v, 10) for v in res['cy']],
        'z'       : [round(v, 10) for v in res['cz']],
    }

with open('polynomial_trajectory.yaml', 'w') as f:
    yaml.dump(ros2_params, f, default_flow_style=False, sort_keys=False)

print(f"Saved 'polynomial_trajectory.yaml'  (ROS 2 parameters, copy to autopilot/config/)")

# ============================================================
#  PLOTTING
# ============================================================

entry_color = 'gray'
circ_colors = plt.cm.tab10(np.linspace(0, 1, N_CIRC))

fig3d = plt.figure(figsize=(10, 6))
ax3d  = fig3d.add_subplot(111, projection='3d')
all_ts = []; all_xyz = {'x':[],'y':[],'z':[]}
all_vel = {'x':[],'y':[],'z':[]}; all_acl = {'x':[],'y':[],'z':[]}
t_offset = 0.0

for i, res in enumerate(solved):
    ts = np.linspace(0.0, res['tf'], 300)
    cx, cy, cz = res['cx'], res['cy'], res['cz']
    xs = np.array([eval_poly(cx, t) for t in ts])
    ys = np.array([eval_poly(cy, t) for t in ts])
    zs = np.array([eval_poly(cz, t) for t in ts])

    all_ts.append(ts + t_offset)
    all_xyz['x'].append(xs); all_xyz['y'].append(ys); all_xyz['z'].append(zs)
    all_vel['x'].append(np.array([eval_vel(cx, t) for t in ts]))
    all_vel['y'].append(np.array([eval_vel(cy, t) for t in ts]))
    all_vel['z'].append(np.array([eval_vel(cz, t) for t in ts]))
    all_acl['x'].append(np.array([eval_acc(cx, t) for t in ts]))
    all_acl['y'].append(np.array([eval_acc(cy, t) for t in ts]))
    all_acl['z'].append(np.array([eval_acc(cz, t) for t in ts]))

    color = entry_color if i < N_ENTRY_SEGMENTS else circ_colors[i - N_ENTRY_SEGMENTS]
    label = 'Entry' if i < N_ENTRY_SEGMENTS else f'Circ {i - N_ENTRY_SEGMENTS + 1}'
    lw    = 1.5 if i < N_ENTRY_SEGMENTS else 2.5
    ax3d.plot(xs, ys, zs, color=color, linewidth=lw, label=label)

    xo, yo, zo = SEGMENTS[i]['obstacle']['pos']
    ax3d.scatter(xo, yo, zo, color='red', s=80, zorder=5)
    ax3d.text(xo, yo, zo, f' O{i}', fontsize=8)
    t_offset += res['tf']

ax3d.scatter(*HOVER_START, color='black', s=120, marker='*', zorder=6, label='Hover start')
ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
ax3d.set_title(f'Entry + {N_CIRC} circular segs  |  ORDER={ORDER}')
ax3d.legend(fontsize=8)
plt.tight_layout()
plt.show(block=False)

# Time-series
ts_cat = np.concatenate(all_ts)
fig, axs_p = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

for key, lbl, c in zip(['x','y','z'], ['x','y','z'], ['tab:blue','tab:orange','tab:green']):
    axs_p[0].plot(ts_cat, np.concatenate(all_xyz[key]), color=c, label=lbl)
    axs_p[1].plot(ts_cat, np.concatenate(all_vel[key]), color=c, label=lbl)
    axs_p[2].plot(ts_cat, np.concatenate(all_acl[key]), color=c, label=lbl)

axs_p[2].axhline(GRAV - MARGIN, color='r', linestyle='--', linewidth=0.8, label='a_z limit')

# Show workspace box limits on the position plot
for val, lbl, col in [(X_MIN,'x_min','tab:blue'), (X_MAX,'x_max','tab:blue'),
                       (Y_MIN,'y_min','tab:orange'), (Y_MAX,'y_max','tab:orange'),
                       (Z_CEIL,'z_ceil','tab:green')]:
    axs_p[0].axhline(val, color=col, linestyle=':', linewidth=0.8, alpha=0.6, label=lbl)

t_b = 0.0
for i, res in enumerate(solved):
    style = ':' if i < N_ENTRY_SEGMENTS else '-.'
    for ax_p in axs_p:
        ax_p.axvline(t_b, color='gray', linestyle=style, linewidth=0.8)
    lbl = 'E' if i < N_ENTRY_SEGMENTS else f'C{i - N_ENTRY_SEGMENTS + 1}'
    axs_p[0].text(t_b + 0.05, axs_p[0].get_ylim()[1], lbl, fontsize=7,
                  color='gray', va='bottom')
    t_b += res['tf']

axs_p[0].set_ylabel('Position [m]');        axs_p[0].legend(); axs_p[0].grid(True)
axs_p[1].set_ylabel('Velocity [m/s]');      axs_p[1].legend(); axs_p[1].grid(True)
axs_p[2].set_ylabel('Acceleration [m/s²]'); axs_p[2].set_xlabel('Time [s]')
axs_p[2].legend(); axs_p[2].grid(True)
fig.suptitle(f'Entry (E) + Circular segs (C1-C{N_CIRC})  — lap-wrap is C2 smooth')
plt.tight_layout()
plt.show()