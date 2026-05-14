#!/usr/bin/env python3
"""
Multi-segment circular trajectory generator — clean C2 + snap baseline.

Test progression:
  Step 1 (this file): seg2 has inclination_deg=90 → vertical wall, should
          behave identically to the other windows. Verify trajectory is clean.
  Step 2: change seg2 inclination_deg to 70 → wall leans 20° forward.
          Verify a clean dive appears without loops.
  Step 3: try 45° and lower once 70° works.
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
ORDER            = 11    # C2: 3 constraints/axis/junction, 12 coeffs → plenty of DOF
N_INT            = 80
M_SAMP           = 60
MARGIN           = 1e-3
GRAV             = 9.81
RHO_X = RHO_Y = RHO_Z = 0.0

MIN_PASS_SPEED = 3.5   # min |v·n| at inclined window [m/s]

HOVER_START = (2.0, -1.0, -1.5)

# ── Segments ──────────────────────────────────────────────────────────────────
# inclination_deg = 90  → vertical wall → constraint = vz=0  (identical to standard)
# approach_dir    = sign of v·n at passage (prevents backward pass through window)
#
# Verification of 90° math:
#   axis='y': n = (0, sin(90°), -cos(90°)) = (0, 1, 0)
#   tangent constraint: vy*cos(90°) + vz*sin(90°) = 0 → vz = 0  ✓

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
    # ── Circ 1  (x-aligned, drone moves in -x direction) ──────────────────────
    {
        'duration' : 3.0,
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
    # ── Circ 2  TEST: inclination_deg=90 (vertical) ───────────────────────────
    # Drone moves in -y direction toward x=-2.
    # approach_dir=-1 because v·n must be negative (n points in +y, v is in -y).
    # Expected: IDENTICAL to setting vy=None, vz=0.
    # Change inclination_deg to 70 for next test step.
    {
        'duration' : 2.5,
        'obstacle' : {
            'pos'              : (-2.0, 0.0, -1.5),
            'phi_deg'          : 20.0,
            'theta_deg'        : 0.0,
            'phi_dot'          : 0.0,
            'theta_dot'        : 0.0,
            'vx'               : 0.0,
            'vy'               : None,
            'vz'               : None,
            'inclination_deg'  : 70.0,   # ← STEP 1: vertical baseline
            'inclination_axis' : 'y',
            'approach_dir'     : -1,     # v·n < 0 (drone moves in -y, n points +y)
        },
    },
    # ── Circ 3  (x-aligned, drone moves in +x direction) ──────────────────────
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
    # ── Circ 4  (y-aligned) ───────────────────────────────────────────────────
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
print(f"Entry: {entry_dur:.2f}s  |  Lap: {circ_dur:.2f}s  |  Total: {total_dur:.2f}s")
print(f"ORDER={ORDER}  n_coeffs={n}  vars={S*(3*n+6)}")

# ============================================================
#  POLYNOMIAL BASIS HELPERS
# ============================================================

def time_power_vec(t, n):
    return np.array([t**i for i in range(n)], dtype=float)

def vel_basis(t, n):
    return np.array([0.0 if i==0 else i*t**(i-1) for i in range(n)], dtype=float)

def acc_basis(t, n):
    return np.array([0.0 if i<=1 else i*(i-1)*t**(i-2) for i in range(n)], dtype=float)

def jerk_basis(t, n):
    return np.array([0.0 if i<=2 else i*(i-1)*(i-2)*t**(i-3) for i in range(n)], dtype=float)

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
    return np.concatenate([sol, np.zeros(max(0, n-len(sol)))])

# ============================================================
#  NLP
# ============================================================

cx_vars  = [ca.SX.sym(f'cx_{i}',n) for i in range(S)]
cy_vars  = [ca.SX.sym(f'cy_{i}',n) for i in range(S)]
cz_vars  = [ca.SX.sym(f'cz_{i}',n) for i in range(S)]
u_vars   = [ca.SX.sym(f'u_{i}')    for i in range(S)]
th_vars  = [ca.SX.sym(f'th_{i}')   for i in range(S)]
ph_vars  = [ca.SX.sym(f'ph_{i}')   for i in range(S)]
ud_vars  = [ca.SX.sym(f'ud_{i}')   for i in range(S)]
thd_vars = [ca.SX.sym(f'thd_{i}')  for i in range(S)]
phd_vars = [ca.SX.sym(f'phd_{i}')  for i in range(S)]
t_sym    = ca.SX.sym('t')

vars_all = ca.vertcat(*[
    v for i in range(S)
    for v in [cx_vars[i],cy_vars[i],cz_vars[i],
              u_vars[i],th_vars[i],ph_vars[i],
              ud_vars[i],thd_vars[i],phd_vars[i]]
])

lbx, ubx = [], []
for _ in range(S):
    lbx += [-ca.inf]*3*n + [1e-6,-np.pi/2,-np.pi/2,-100.,-50.,-50.]
    ubx += [ ca.inf]*3*n + [1e3,  np.pi/2, np.pi/2, 100., 50., 50.]

total_cost = 0
g_cons, lbg, ubg = [], [], []

def eq(expr, val):
    g_cons.append(expr); lbg.append(float(val)); ubg.append(float(val))

def ineq_upper(expr, ub):
    g_cons.append(expr); lbg.append(-ca.inf); ubg.append(float(ub))

def ineq_lower(expr, lb):
    g_cons.append(expr); lbg.append(float(lb)); ubg.append(ca.inf)

hx, hy, hz = HOVER_START

# Hover start: pos + vel + acc = 0
for c_vec, val in zip([cx_vars[0],cy_vars[0],cz_vars[0]], [hx,hy,hz]):
    eq(ca.dot(c_vec, ca.DM(time_power_vec(0.0,n))), val)
    eq(ca.dot(c_vec, ca.DM(vel_basis(0.0,n))),      0.0)
    eq(ca.dot(c_vec, ca.DM(acc_basis(0.0,n))),      0.0)

# Lap wrap C2
tf_last = float(SEGMENTS[-1]['duration'])
for c_last, c_first in zip([cx_vars[-1],cy_vars[-1],cz_vars[-1]],
                            [cx_vars[first_c],cy_vars[first_c],cz_vars[first_c]]):
    for basis in [time_power_vec, vel_basis, acc_basis]:
        eq(ca.dot(c_last,  ca.DM(basis(tf_last,n))) -
           ca.dot(c_first, ca.DM(basis(0.0,   n))), 0.0)

# Per-segment
for i, seg in enumerate(SEGMENTS):
    cx=cx_vars[i]; cy=cy_vars[i]; cz=cz_vars[i]
    u_s=u_vars[i]; th_s=th_vars[i]; ph_s=ph_vars[i]
    ud_s=ud_vars[i]; thd_s=thd_vars[i]; phd_s=phd_vars[i]

    tf    = float(seg['duration'])
    t_mid = tf/2.0
    obs   = seg['obstacle']
    xm,ym,zm = obs['pos']

    inc_deg  = obs.get('inclination_deg', None)
    inc_axis = obs.get('inclination_axis','x')

    if inc_deg is not None:
        alpha = np.deg2rad(float(inc_deg))
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        if inc_axis == 'x':
            phi_des   = np.deg2rad(float(obs.get('phi_deg',   0.0)))
            theta_des = np.deg2rad(float(obs.get('theta_deg', inc_deg-90.0)))
        else:
            phi_des   = np.deg2rad(float(obs.get('phi_deg',   90.0-inc_deg)))
            theta_des = np.deg2rad(float(obs.get('theta_deg', 0.0)))
    else:
        phi_des   = np.deg2rad(float(obs.get('phi_deg',  0.0)))
        theta_des = np.deg2rad(float(obs.get('theta_deg',0.0)))
        alpha = cos_a = sin_a = None

    phi_dot_des   = float(obs.get('phi_dot',  0.0))
    theta_dot_des = float(obs.get('theta_dot',0.0))

    # Snap cost
    snx = sum(j*(j-1)*(j-2)*(j-3)*cx[j]*t_sym**(j-4) for j in range(4,n))
    sny = sum(j*(j-1)*(j-2)*(j-3)*cy[j]*t_sym**(j-4) for j in range(4,n))
    snz = sum(j*(j-1)*(j-2)*(j-3)*cz[j]*t_sym**(j-4) for j in range(4,n))
    dt_int = tf/float(N_INT)
    for k in range(N_INT+1):
        ti = k*dt_int
        total_cost += ca.substitute(snx**2+sny**2+snz**2, t_sym, ti)*dt_int

    # C2 junction continuity
    if i < S-1:
        cx_n=cx_vars[i+1]; cy_n=cy_vars[i+1]; cz_n=cz_vars[i+1]
        for c_c,c_n in zip([cx,cy,cz],[cx_n,cy_n,cz_n]):
            for basis in [time_power_vec, vel_basis, acc_basis]:
                eq(ca.dot(c_c,ca.DM(basis(tf, n))) -
                   ca.dot(c_n,ca.DM(basis(0.0,n))), 0.0)

    # Obstacle position
    eq(ca.dot(cx,ca.DM(time_power_vec(t_mid,n))), xm)
    eq(ca.dot(cy,ca.DM(time_power_vec(t_mid,n))), ym)
    eq(ca.dot(cz,ca.DM(time_power_vec(t_mid,n))), zm)

    # Explicit velocity constraints
    for c_vec,v_des in zip([cx,cy,cz],[obs.get('vx'),obs.get('vy'),obs.get('vz')]):
        if v_des is not None:
            eq(ca.dot(c_vec,ca.DM(vel_basis(t_mid,n))), float(v_des))

    # Inclination direction constraint
    # At 90°: cos(90°)=0, sin(90°)=1 → constraint = vz=0 ✓
    if inc_deg is not None:
        vx_s = ca.dot(cx,ca.DM(vel_basis(t_mid,n)))
        vy_s = ca.dot(cy,ca.DM(vel_basis(t_mid,n)))
        vz_s = ca.dot(cz,ca.DM(vel_basis(t_mid,n)))
        if inc_axis == 'x':
            eq(vx_s*cos_a + vz_s*sin_a, 0.0)
            v_normal = vx_s*np.sin(alpha) - vz_s*np.cos(alpha)
        else:
            eq(vy_s*cos_a + vz_s*sin_a, 0.0)
            v_normal = vy_s*np.sin(alpha) - vz_s*np.cos(alpha)
        approach_dir = float(obs.get('approach_dir',1.0))
        ineq_lower(approach_dir*v_normal, MIN_PASS_SPEED)

    # NED coupling
    ax_m=ca.dot(cx,ca.DM(acc_basis(t_mid,n))); ay_m=ca.dot(cy,ca.DM(acc_basis(t_mid,n)))
    az_m=ca.dot(cz,ca.DM(acc_basis(t_mid,n))); vx_m=ca.dot(cx,ca.DM(vel_basis(t_mid,n)))
    vy_m=ca.dot(cy,ca.DM(vel_basis(t_mid,n))); vz_m=ca.dot(cz,ca.DM(vel_basis(t_mid,n)))
    fdx=ca.DM(RHO_X)*vx_m; fdy=ca.DM(RHO_Y)*vy_m; fdz=ca.DM(RHO_Z)*vz_m
    fdot_x=ca.DM(RHO_X)*ax_m; fdot_y=ca.DM(RHO_Y)*ay_m; fdot_z=ca.DM(RHO_Z)*az_m

    eq((-ax_m-fdx)-u_s*ca.sin(th_s)*ca.cos(ph_s), 0.0)
    eq((-ay_m-fdy)-u_s*ca.sin(ph_s),               0.0)
    eq((GRAV-az_m-fdz)-u_s*ca.cos(ph_s)*ca.cos(th_s), 0.0)
    eq(ph_s, phi_des); eq(th_s, theta_des)

    jx_c=-ca.dot(cx,ca.DM(jerk_basis(t_mid,n))); jy_c=-ca.dot(cy,ca.DM(jerk_basis(t_mid,n)))
    jz_c=-ca.dot(cz,ca.DM(jerk_basis(t_mid,n)))
    bx_dot=jx_c-fdot_x; by_dot=jy_c-fdot_y; bz_dot=jz_c-fdot_z

    s_th=ca.sin(th_s); c_th=ca.cos(th_s); s_ph=ca.sin(ph_s); c_ph=ca.cos(ph_s)
    A11=s_th*c_ph; A12=u_s*c_th*c_ph; A13=-u_s*s_th*s_ph
    A21=-s_ph;      A22=0.0;            A23=-u_s*c_ph
    A31=c_th*c_ph; A32=-u_s*c_ph*s_th; A33=-u_s*s_ph*c_th
    eq(A11*ud_s+A12*thd_s+A13*phd_s-bx_dot, 0.0)
    eq(A21*ud_s+A22*thd_s+A23*phd_s-by_dot, 0.0)
    eq(A31*ud_s+A32*thd_s+A33*phd_s-bz_dot, 0.0)
    eq(phd_s, phi_dot_des); eq(thd_s, theta_dot_des)

    # No negative thrust
    for k in range(M_SAMP):
        ti = (k/float(max(1,M_SAMP-1)))*tf
        ineq_upper(ca.dot(cz,ca.DM(acc_basis(ti,n))), GRAV-MARGIN)

# ============================================================
#  INITIAL GUESS
# ============================================================

x0_list = []
for i, seg in enumerate(SEGMENTS):
    tf=float(seg['duration']); t_mid=tf/2.0
    obs=seg['obstacle']; xm,ym,zm=obs['pos']
    xs,ys,zs = (hx,hy,hz) if i==0 else SEGMENTS[i-1]['obstacle']['pos']
    xe,ye,ze = SEGMENTS[i+1]['obstacle']['pos'] if i<S-1 else SEGMENTS[first_c]['obstacle']['pos']
    cx0=fit_initial_coeffs([('pos',0.,xs),('pos',t_mid,xm),('pos',tf,xe)],n)
    cy0=fit_initial_coeffs([('pos',0.,ys),('pos',t_mid,ym),('pos',tf,ye)],n)
    cz0=fit_initial_coeffs([('pos',0.,zs),('pos',t_mid,zm),('pos',tf,ze)],n)
    inc_deg=obs.get('inclination_deg'); inc_axis=obs.get('inclination_axis','x')
    if inc_deg is not None:
        theta_g=float(obs.get('theta_deg', inc_deg-90. if inc_axis=='x' else 0.))
        phi_g  =float(obs.get('phi_deg',   0.          if inc_axis=='x' else 90.-inc_deg))
    else:
        theta_g=float(obs.get('theta_deg',0.)); phi_g=float(obs.get('phi_deg',0.))
    x0_list.append(np.concatenate([cx0,cy0,cz0,
        [GRAV,np.deg2rad(theta_g),np.deg2rad(phi_g),
         0.,float(obs.get('theta_dot',0.)),float(obs.get('phi_dot',0.))]]))
x0_full = np.concatenate(x0_list)

# ============================================================
#  SOLVE
# ============================================================

print(f"Constraints: {len(g_cons)}\n")
nlp  = {'x':vars_all,'f':total_cost,'g':ca.vertcat(*g_cons)}
opts = {'ipopt.print_level':3,'print_time':True,'ipopt.max_iter':3000,'ipopt.tol':1e-6}
solver = ca.nlpsol('solver','ipopt',nlp,opts)
sol    = solver(x0=x0_full.tolist(),lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg)
solx   = sol['x'].full().flatten()
print(f"\nSolver status: {solver.stats()['return_status']}")

# ============================================================
#  EXTRACT + DIAGNOSTICS
# ============================================================

n_per=3*n+6; solved=[]
print("\n── Results ──────────────────────────────────────────────────────────")
for i in range(S):
    blk=solx[i*n_per:(i+1)*n_per]
    cx_sol=blk[0:n]; cy_sol=blk[n:2*n]; cz_sol=blk[2*n:3*n]
    u_s,th_s,ph_s=blk[3*n],blk[3*n+1],blk[3*n+2]
    tf=float(SEGMENTS[i]['duration']); t_mid=tf/2.0; obs=SEGMENTS[i]['obstacle']
    vx_m=eval_vel(cx_sol,t_mid); vy_m=eval_vel(cy_sol,t_mid); vz_m=eval_vel(cz_sol,t_mid)
    v_mag=np.sqrt(vx_m**2+vy_m**2+vz_m**2)
    label='ENTRY' if i<N_ENTRY_SEGMENTS else f'CIRC-{i-N_ENTRY_SEGMENTS+1}'
    inc_deg=obs.get('inclination_deg'); inc_str=f' [inc={inc_deg}°]' if inc_deg else ''
    print(f"  seg{i}[{label}]{inc_str}: θ={np.rad2deg(th_s):.1f}°  φ={np.rad2deg(ph_s):.1f}°  "
          f"vel=({vx_m:.2f},{vy_m:.2f},{vz_m:.2f})  |v|={v_mag:.2f}")
    if inc_deg is not None:
        alpha=np.deg2rad(float(inc_deg)); inc_axis=obs.get('inclination_axis','x')
        if inc_axis=='x':
            n_w=np.array([np.sin(alpha),0.,-np.cos(alpha)]); tang_r=vx_m*np.cos(alpha)+vz_m*np.sin(alpha)
        else:
            n_w=np.array([0.,np.sin(alpha),-np.cos(alpha)]); tang_r=vy_m*np.cos(alpha)+vz_m*np.sin(alpha)
        n_w/=np.linalg.norm(n_w); v_n=np.dot([vx_m,vy_m,vz_m],n_w)
        print(f"    n={np.round(n_w,3)}  tang_res={tang_r:.4f}  v·n={v_n:.3f}  "
              f"angle={np.rad2deg(np.arctan2(abs(tang_r),abs(v_n))):.1f}°")
    solved.append({'cx':cx_sol.tolist(),'cy':cy_sol.tolist(),'cz':cz_sol.tolist(),'tf':tf,'t_mid':t_mid})

print("\n── Lap-wrap C2 check ───────────────────────────────────────────────")
last=solved[-1]; fcs=solved[first_c]
for axis,cl,cf in zip(['x','y','z'],[last['cx'],last['cy'],last['cz']],[fcs['cx'],fcs['cy'],fcs['cz']]):
    print(f"  {axis}: Δpos={eval_poly(cl,last['tf'])-eval_poly(cf,0.):.2e}  "
          f"Δvel={eval_vel(cl,last['tf'])-eval_vel(cf,0.):.2e}  "
          f"Δacc={eval_acc(cl,last['tf'])-eval_acc(cf,0.):.2e}")

# ============================================================
#  YAML
# ============================================================

ros2={'/**':{'ros__parameters':{'autopilot':{'polynomial_trajectory':{
    'num_laps':N_LAPS,'num_entry_segments':N_ENTRY_SEGMENTS,
    'num_circular_segments':N_CIRC,'num_segments_total':S,
    'entry_duration':round(entry_dur,6),'circular_lap_duration':round(circ_dur,6),
    'drag':{'rho_x':RHO_X,'rho_y':RHO_Y,'rho_z':RHO_Z},'segments':{}}}}}}
sd=ros2['/**']['ros__parameters']['autopilot']['polynomial_trajectory']['segments']
for i,res in enumerate(solved):
    obs=SEGMENTS[i]['obstacle']
    inc=float(obs.get('inclination_deg',90.)); iax=str(obs.get('inclination_axis','x'))
    if obs.get('inclination_deg') is not None:
        th_o=float(obs.get('theta_deg',0. if iax=='y' else inc-90.))
        ph_o=float(obs.get('phi_deg',  90.-inc if iax=='y' else 0.))
    else:
        th_o=float(obs.get('theta_deg',0.)); ph_o=float(obs.get('phi_deg',0.))
    sd[f'seg{i}']={'duration':round(res['tf'],6),
        'x':[round(v,10) for v in res['cx']],'y':[round(v,10) for v in res['cy']],
        'z':[round(v,10) for v in res['cz']],
        'obstacle':{'pos':[round(v,6) for v in obs['pos']],'phi_deg':round(ph_o,4),
                    'theta_deg':round(th_o,4),'inclination_deg':round(inc,4),'inclination_axis':iax}}
with open('polynomial_trajectory.yaml','w') as f: yaml.dump(ros2,f,default_flow_style=False,sort_keys=False)
print("\nSaved 'polynomial_trajectory.yaml'")

# ============================================================
#  PLOT
# ============================================================

WINDOW_SIZE = 0.4

def draw_window_3d(ax, obs, color, idx):
    xm,ym,zm=obs['pos']; center=np.array([xm,ym,zm])
    inc=float(obs.get('inclination_deg',90.)); iax=obs.get('inclination_axis','x')
    ph=np.deg2rad(float(obs.get('phi_deg',0.))); alpha=np.deg2rad(inc)
    if iax=='x':
        normal=np.array([np.sin(alpha),0.,-np.cos(alpha)]); e1=np.array([0.,1.,0.]); e2=np.array([np.cos(alpha),0.,np.sin(alpha)])
    else:
        normal=np.array([0.,np.sin(alpha),-np.cos(alpha)]); e1=np.array([1.,0.,0.]); e2=np.array([0.,np.cos(alpha),np.sin(alpha)])
    normal/=np.linalg.norm(normal)
    e1r=np.cos(ph)*e1+np.sin(ph)*e2; e2r=-np.sin(ph)*e1+np.cos(ph)*e2
    hw=WINDOW_SIZE/2.
    corners=np.array([center+hw*e1r+hw*e2r,center-hw*e1r+hw*e2r,center-hw*e1r-hw*e2r,center+hw*e1r-hw*e2r])
    ax.add_collection3d(Poly3DCollection([corners],alpha=0.35,facecolor=color,edgecolor=color,linewidth=1.8))
    al=0.35
    ax.quiver(xm,ym,zm,normal[0]*al,normal[1]*al,normal[2]*al,color='gold',linewidth=2.,arrow_length_ratio=0.35)
    ax.text(xm+normal[0]*0.08,ym+normal[1]*0.08,zm+normal[2]*0.08,f' W{idx}',fontsize=8,color=color)

ec='gray'; cc=plt.cm.tab10(np.linspace(0,1,N_CIRC))
fig3d=plt.figure(figsize=(10,7)); ax3d=fig3d.add_subplot(111,projection='3d')
t_off=0.
for i,res in enumerate(solved):
    ts=np.linspace(0.,res['tf'],300); cx,cy,cz=res['cx'],res['cy'],res['cz']
    xs=np.array([eval_poly(cx,t) for t in ts]); ys=np.array([eval_poly(cy,t) for t in ts]); zs=np.array([eval_poly(cz,t) for t in ts])
    color=ec if i<N_ENTRY_SEGMENTS else cc[i-N_ENTRY_SEGMENTS]
    lbl='Entry' if i<N_ENTRY_SEGMENTS else f'C{i-N_ENTRY_SEGMENTS+1}'
    ax3d.plot(xs,ys,zs,color=color,linewidth=2.,label=lbl)
    draw_window_3d(ax3d,SEGMENTS[i]['obstacle'],color,i)
    t_off+=res['tf']
ax3d.scatter(*HOVER_START,color='black',s=120,marker='*',label='Start')
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.set_title(f'C2+snap  ORDER={ORDER}  — STEP1: seg2 incl=90° (vertical baseline)')
ax3d.legend(fontsize=8); plt.tight_layout(); plt.show(block=False)

# Time-series
all_ts=[]; all_p={'x':[],'y':[],'z':[]}; all_v={'x':[],'y':[],'z':[]}; all_a={'x':[],'y':[],'z':[]}
t_off=0.
for res in solved:
    ts=np.linspace(0.,res['tf'],300); cx,cy,cz=res['cx'],res['cy'],res['cz']
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
    t_off+=res['tf']
ts_c=np.concatenate(all_ts)
fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
for k,l,c in zip(['x','y','z'],['x','y','z'],['tab:blue','tab:orange','tab:green']):
    axs[0].plot(ts_c,np.concatenate(all_p[k]),color=c,label=l)
    axs[1].plot(ts_c,np.concatenate(all_v[k]),color=c,label=l)
    axs[2].plot(ts_c,np.concatenate(all_a[k]),color=c,label=l)
axs[2].axhline(GRAV-MARGIN,color='r',linestyle='--',linewidth=0.8)
t_b=0.
for res in solved:
    for ax in axs: ax.axvline(t_b,color='gray',linestyle=':',linewidth=0.8)
    t_b+=res['tf']
axs[0].set_ylabel('Pos [m]'); axs[0].legend(); axs[0].grid(True)
axs[1].set_ylabel('Vel [m/s]'); axs[1].legend(); axs[1].grid(True)
axs[2].set_ylabel('Acc [m/s²]'); axs[2].set_xlabel('Time [s]'); axs[2].legend(); axs[2].grid(True)
plt.tight_layout(); plt.show()