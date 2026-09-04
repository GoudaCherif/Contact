import time
debut = time.perf_counter()

import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile
import ufl
from petsc4py import PETSc
import os
import sys

#  Tags physiques 
TAG_OS       = 1
TAG_IMPL     = 2
TAG_GAP      = 3
TAG_AXE_OS   = 1
TAG_EXT      = 2
TAG_BOT      = 3
TAG_INT_OS   = 4
TAG_TOP      = 5
TAG_INT_IMPL = 6
TAG_AXE_IMPL = 7

save_dir = os.path.dirname(os.path.abspath(__file__))

with XDMFFile(MPI.COMM_WORLD,
              os.path.join(save_dir, "mesh_cones.xdmf"), "r") as xdmf:
    domain = xdmf.read_mesh()
    domain.topology.create_entities(1)
    cell_tags  = xdmf.read_meshtags(domain, name="cell_tags")
    facet_tags = xdmf.read_meshtags(domain, name="facet_tags")

print(f"Mesh — Cellules : {domain.topology.index_map(2).size_global} | "
      f"Sommets : {domain.topology.index_map(0).size_global}")
coords = domain.geometry.x
print(f"  z min : {coords[:,1].min():.3f} mm  z max : {coords[:,1].max():.3f} mm")
sys.stdout.flush()

#  Paramètres
E_r       = 822.0
E_z       = 1352.0
nu_rtheta = 0.3
nu_rz     = 0.3
G_rz      = 399.0
nu_zr     = nu_rz * E_z / E_r
E_2       = 113000.0
nu_2      = 0.3
gamma     = 1e-4
mu_m      = 1.0
u_imposed = -1.7
n_steps   = 20
warp_factor = 1

#  Espace P1 
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
u = fem.Function(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
r = x[0]
print(f"DOFs : {V.dofmap.index_map.size_global * 2}")
sys.stdout.flush()

# Cinématique 
def deformation_gradient_axi(u, r):
    F_2d = ufl.Identity(2) + ufl.grad(u)
    F_tt = 1 + u[0] / r
    return ufl.as_matrix([
        [F_2d[0,0], 0,    F_2d[0,1]],
        [0,         F_tt, 0        ],
        [F_2d[1,0], 0,    F_2d[1,1]]
    ])

def strain_axi(u, r):
    return ufl.as_vector([
        u[0].dx(0),
        u[0] / r,
        u[1].dx(1),
        0.5 * (u[0].dx(1) + u[1].dx(0)) * 2
    ])

def C_isotrope(E, nu):
    lmbda = E * nu / ((1 + nu) * (1 - 2*nu))
    mu    = E / (2 * (1 + nu))
    return ufl.as_matrix([
        [lmbda+2*mu, lmbda,      lmbda,      0 ],
        [lmbda,      lmbda+2*mu, lmbda,      0 ],
        [lmbda,      lmbda,      lmbda+2*mu, 0 ],
        [0,          0,          0,          mu]
    ])

def C_transverse_isotrope(E_r, E_z, nu_rtheta, nu_rz, G_rz):
    nu_zr = nu_rz * E_z / E_r
    S = ufl.as_matrix([
        [ 1/E_r,         -nu_rtheta/E_r, -nu_zr/E_z, 0      ],
        [-nu_rtheta/E_r,  1/E_r,         -nu_zr/E_z, 0      ],
        [-nu_rz/E_r,     -nu_rz/E_r,      1/E_z,      0      ],
        [ 0,              0,               0,          1/G_rz ]
    ])
    return ufl.inv(S)

C_os   = C_transverse_isotrope(E_r, E_z, nu_rtheta, nu_rz, G_rz)
C_impl = C_isotrope(E_2, nu_2)

#  Énergies 
def W_lineaire(u, r, C):
    eps = strain_axi(u, r)
    return 0.5 * ufl.dot(eps, C * eps)

def W_neohookean_iso(u, r, mu, gamma):
    F  = deformation_gradient_axi(u, r)
    C  = F.T * F
    Ic = ufl.tr(C)
    J  = ufl.det(F)
    W_iso = (mu / 2) * (J**(-2/3) * Ic - 3)
    return gamma * W_iso

dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

W_total = (
    W_lineaire(u, r, C_os)              * r * dx(TAG_OS)   +
    W_lineaire(u, r, C_impl)            * r * dx(TAG_IMPL) +
    W_neohookean_iso(u, r, mu_m, gamma) * r * dx(TAG_GAP)
)

F_form = ufl.derivative(W_total, u, v)
J_form = ufl.derivative(F_form, u, ufl.TrialFunction(V))
print("Formes variationnelles définies (sans régularisation).")
sys.stdout.flush()

# Conditions aux limites 
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

bot_facets  = facet_tags.find(TAG_BOT)
top_facets  = facet_tags.find(TAG_TOP)
axe_facets  = np.concatenate([
    facet_tags.find(TAG_AXE_OS),
    facet_tags.find(TAG_AXE_IMPL)
])

dofs_bot_r = fem.locate_dofs_topological(V.sub(0), fdim, bot_facets)
dofs_bot_z = fem.locate_dofs_topological(V.sub(1), fdim, bot_facets)
dofs_axe_r = fem.locate_dofs_topological(V.sub(0), fdim, axe_facets)
dofs_top_z = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)

zero  = fem.Constant(domain, PETSc.ScalarType(0.0))
u_imp = fem.Constant(domain, PETSc.ScalarType(0.0))

bcs = [
    fem.dirichletbc(zero,  dofs_bot_r, V.sub(0)),
    fem.dirichletbc(zero,  dofs_bot_z, V.sub(1)),
    fem.dirichletbc(zero,  dofs_axe_r, V.sub(0)),
    fem.dirichletbc(u_imp, dofs_top_z, V.sub(1)),
]
print(f"CL — BOT:{len(dofs_bot_z)} AXE:{len(dofs_axe_r)} TOP:{len(dofs_top_z)}")
sys.stdout.flush()

from dolfinx.mesh import entities_to_geometry
top_nodes = np.unique(entities_to_geometry(domain, 1, top_facets, False).flatten())
print(f"z noeuds TOP : min={domain.geometry.x[top_nodes,1].min():.3f}  "
      f"max={domain.geometry.x[top_nodes,1].max():.3f}")

#  Solveur 
from dolfinx.fem.petsc import NonlinearProblem

def make_problem():
    return NonlinearProblem(
        F_form, u, bcs=bcs, J=J_form,
        petsc_options={
            "snes_type"                 : "newtonls",
            "snes_linesearch_type"      : "bt",
            "snes_atol"                 : 1e-9,
            "snes_rtol"                 : 1e-9,
            "snes_max_it"               : 100,
            "snes_monitor"              : None,
            "snes_converged_reason"     : None,
            "ksp_type"                  : "preonly",
            "pc_type"                   : "lu",
            "pc_factor_mat_solver_type" : "mumps",
            "pc_factor_mat_solver_package" : "mumps",
            "mat_mumps_icntl_14"        : 200,
            "mat_mumps_cntl_1"          : 0.1,
        },
        petsc_options_prefix="tmc"
    )

problem = make_problem()

#  Post‑traitement 
W0 = fem.functionspace(domain, ("DG", 0))
J_expr = fem.Expression(ufl.det(deformation_gradient_axi(u, r)),
                         W0.element.interpolation_points)
J_field = fem.Function(W0)
cells_gap  = np.where(cell_tags.values == TAG_GAP)[0]
cells_os   = np.where(cell_tags.values == TAG_OS)[0]
cells_impl = np.where(cell_tags.values == TAG_IMPL)[0]

cells_impl_nodes = np.unique(
    entities_to_geometry(domain, 2, cells_impl, False).flatten()
)

node_top_axis = top_nodes[np.argmin(coords[top_nodes, 0])]
print(f"Point suivi (haut implant, axe) — r={coords[node_top_axis,0]:.4f} mm, "
      f"z={coords[node_top_axis,1]:.4f} mm")

hist_steps     = []
hist_j_min     = []
hist_j_max     = []
hist_uz_min    = []
hist_uz_impl   = []
hist_uz_top    = []
hist_step_num  = []
snapshots      = []

u_per_step_initial = u_imposed / n_steps
u_per_step = u_per_step_initial
current_u  = 0.0
step_count = 0
attempt_count = 0
max_attempts  = 500
delta_min = 1e-6 * abs(u_imposed)

u_prev = u.x.array.copy()

print(f"Résolution — pas initial de {u_per_step:.4f} mm (max {n_steps} pas)")
sys.stdout.flush()

while abs(current_u) < abs(u_imposed) and attempt_count < max_attempts:
    target_u = current_u + u_per_step
    if abs(target_u) > abs(u_imposed):
        target_u = u_imposed
        u_per_step = target_u - current_u

    u_imp.value = target_u
    attempt_count += 1
    print(f"Tentative {attempt_count} : u={target_u:.4f} mm (incr={u_per_step:.4f})")
    sys.stdout.flush()

    try:
        converged = problem.solve()
        reason = problem.solver.getConvergedReason()
        if reason > 0:
            current_u = target_u
            step_count += 1
            u_prev = u.x.array.copy()
            print(f"  OK (reason={reason})")

            if step_count in [1, n_steps//4, n_steps//2, n_steps]:
                u_snapshot = u.x.array.reshape(-1, 2).copy()
                snapshots.append((current_u, u_snapshot))

            J_field.interpolate(J_expr)
            j_min = J_field.x.array[cells_gap].min()
            j_max = J_field.x.array[cells_gap].max()
            u_arr = u.x.array.reshape(-1, 2)
            uz_min = u_arr[:,1].min()
            uz_impl = u_arr[cells_impl_nodes, 1].mean()
            uz_top  = u_arr[node_top_axis, 1]
            hist_steps.append(current_u)
            hist_j_min.append(j_min)
            hist_j_max.append(j_max)
            hist_uz_min.append(uz_min)
            hist_uz_impl.append(uz_impl)
            hist_uz_top.append(uz_top)
            hist_step_num.append(step_count)

            print(f"  Step {step_count:3d} | u={current_u:.3f} mm | "
                  f"u_z_min={uz_min:.4f} | u_z_impl={uz_impl:.4f} | "
                  f"u_z_top={uz_top:.4f} | "
                  f"J=[{j_min:.4f},{j_max:.4f}]")
            sys.stdout.flush()

        else:
            u.x.array[:] = u_prev
            print(f"  Échec (reason={reason}), restauration + réduction du pas")
            sys.stdout.flush()
            u_per_step /= 2.0
            if abs(u_per_step) < delta_min:
                print("  Pas trop petit, arrêt.")
                break
            problem = make_problem()
            continue

    except Exception as e:
        u.x.array[:] = u_prev
        print(f"  Exception : {e}, restauration + réduction du pas")
        sys.stdout.flush()
        u_per_step /= 2.0
        if abs(u_per_step) < delta_min:
            print("  Pas trop petit, arrêt.")
            break
        problem = make_problem()
        continue

print(f"Résolution terminée après {step_count} pas convergés (sur {n_steps} initialement prévus).")
sys.stdout.flush()

u_array = u.x.array.reshape(-1, 2)
eps_zz = fem.Function(W0)
eps_zz.interpolate(fem.Expression(u[1].dx(1), W0.element.interpolation_points))
J_field.interpolate(J_expr)

idx_jmin_gap = cells_gap[np.argmin(J_field.x.array[cells_gap])]
jmin_nodes   = entities_to_geometry(domain, 2,
                                    np.array([idx_jmin_gap]), False).flatten()
jmin_center  = coords[jmin_nodes].mean(axis=0)
print(f"Cellule J_min — centre : r={jmin_center[0]:.3f} mm, z={jmin_center[1]:.3f} mm")

print(f"u_z — min:{u_array[:,1].min():.4f}  max:{u_array[:,1].max():.4f} mm")
print(f"u_r — min:{u_array[:,0].min():.4f}  max:{u_array[:,0].max():.4f} mm")
print(f"J gap  — min:{J_field.x.array[cells_gap].min():.4f}  "
      f"max:{J_field.x.array[cells_gap].max():.4f}")
print(f"ε_zz os      : {eps_zz.x.array[cells_os].mean():.4e}")
print(f"ε_zz gap     : {eps_zz.x.array[cells_gap].mean():.4e}")
print(f"ε_zz implant : {eps_zz.x.array[cells_impl].mean():.4e}")

mask_axe = np.isclose(coords[:,0], 0.0, atol=0.01)
print(f"Axe u_r — min:{u_array[mask_axe,0].min():.6f}  "
      f"max:{u_array[mask_axe,0].max():.6f}")
print(f"Axe u_z — min:{u_array[mask_axe,1].min():.4f}  "
      f"max:{u_array[mask_axe,1].max():.4f}")

idx_jmax_gap = cells_gap[np.argmax(J_field.x.array[cells_gap])]
jmax_nodes   = entities_to_geometry(domain, 2,
                                    np.array([idx_jmax_gap]), False).flatten()
jmax_center  = coords[jmax_nodes].mean(axis=0)
print(f"Cellule J_max — centre : r={jmax_center[0]:.3f} mm, "
      f"z={jmax_center[1]:.3f} mm")

eps_rr_expr = u[0].dx(0)
eps_tt_expr = u[0] / r
eps_zz_expr = u[1].dx(1)
eps_rz_expr = 0.5 * (u[0].dx(1) + u[1].dx(0))

eps_rr_field = fem.Function(W0)
eps_tt_field = fem.Function(W0)
eps_zz_field = fem.Function(W0)
eps_rz_field = fem.Function(W0)

eps_rr_field.interpolate(fem.Expression(eps_rr_expr, W0.element.interpolation_points))
eps_tt_field.interpolate(fem.Expression(eps_tt_expr, W0.element.interpolation_points))
eps_zz_field.interpolate(fem.Expression(eps_zz_expr, W0.element.interpolation_points))
eps_rz_field.interpolate(fem.Expression(eps_rz_expr, W0.element.interpolation_points))

trace_eps = eps_rr_expr + eps_tt_expr + eps_zz_expr
dev_rr = eps_rr_expr - trace_eps / 3
dev_tt = eps_tt_expr - trace_eps / 3
dev_zz = eps_zz_expr - trace_eps / 3
dev_rz = eps_rz_expr
I2_dev_expr = 0.5 * (dev_rr**2 + dev_tt**2 + dev_zz**2 + 2 * dev_rz**2)

I2_dev_field = fem.Function(W0)
I2_dev_field.interpolate(fem.Expression(I2_dev_expr, W0.element.interpolation_points))

distorsion_expr = ufl.sqrt(3 * I2_dev_expr)
distorsion_field = fem.Function(W0)
distorsion_field.interpolate(fem.Expression(distorsion_expr, W0.element.interpolation_points))

eps_rr_gap = eps_rr_field.x.array[cells_gap]
eps_tt_gap = eps_tt_field.x.array[cells_gap]
eps_zz_gap = eps_zz_field.x.array[cells_gap]
eps_rz_gap = eps_rz_field.x.array[cells_gap]
I2_dev_gap = I2_dev_field.x.array[cells_gap]
dist_gap   = distorsion_field.x.array[cells_gap]

print("\n--- Statistiques de distorsion dans le gap ---")
print(f"eps_rr : min={eps_rr_gap.min():.4e}  max={eps_rr_gap.max():.4e}  mean={eps_rr_gap.mean():.4e}")
print(f"eps_tt : min={eps_tt_gap.min():.4e}  max={eps_tt_gap.max():.4e}  mean={eps_tt_gap.mean():.4e}")
print(f"eps_zz : min={eps_zz_gap.min():.4e}  max={eps_zz_gap.max():.4e}  mean={eps_zz_gap.mean():.4e}")
print(f"eps_rz : min={eps_rz_gap.min():.4e}  max={eps_rz_gap.max():.4e}  mean={eps_rz_gap.mean():.4e}")
print(f"I2_dev : min={I2_dev_gap.min():.4e}  max={I2_dev_gap.max():.4e}  mean={I2_dev_gap.mean():.4e}")
print(f"Distorsion : min={dist_gap.min():.4e}  max={dist_gap.max():.4e}  mean={dist_gap.mean():.4e}")

print("\n--- Contraintes de Von Mises et déplacements (modèle complet) ---")

V_scal = fem.functionspace(domain, ("Lagrange", 1))
u_r_field = fem.Function(V_scal)
u_z_field = fem.Function(V_scal)
u_r_field.name = "u_r"
u_z_field.name = "u_z"
u_r_field.interpolate(fem.Expression(u[0], V_scal.element.interpolation_points))
u_z_field.interpolate(fem.Expression(u[1], V_scal.element.interpolation_points))

V_warp = fem.functionspace(domain, ("Lagrange", 1, (3,)))
u_warp_field = fem.Function(V_warp)
u_warp_field.name = "u_warp"
zero_scalar = fem.Constant(domain, PETSc.ScalarType(0.0))
u_warp_field.interpolate(fem.Expression(
    ufl.as_vector([u[0], u[1], zero_scalar]),
    V_warp.element.interpolation_points))

def sigma_voigt_lin(u, r, C):
    eps = strain_axi(u, r)
    return C * eps

sig_os   = sigma_voigt_lin(u, r, C_os)
sig_impl = sigma_voigt_lin(u, r, C_impl)

F_gap   = deformation_gradient_axi(u, r)
Fv      = ufl.variable(F_gap)
Cg      = Fv.T * Fv
Ic_g    = ufl.tr(Cg)
Jg      = ufl.det(Fv)
W_iso_g = gamma * (mu_m / 2) * (Jg**(-2/3) * Ic_g - 3)
P_gap   = ufl.diff(W_iso_g, Fv)
sigma_gap_tensor = (1 / Jg) * P_gap * Fv.T

sig_gap_rr = sigma_gap_tensor[0, 0]
sig_gap_tt = sigma_gap_tensor[1, 1]
sig_gap_zz = sigma_gap_tensor[2, 2]
sig_gap_rz = sigma_gap_tensor[0, 2]

def von_mises_axi(s_rr, s_tt, s_zz, s_rz):
    return ufl.sqrt(0.5 * ((s_rr - s_tt)**2 + (s_tt - s_zz)**2 + (s_zz - s_rr)**2)
                     + 3 * s_rz**2)

vm_os_expr   = von_mises_axi(sig_os[0],   sig_os[1],   sig_os[2],   sig_os[3])
vm_impl_expr = von_mises_axi(sig_impl[0], sig_impl[1], sig_impl[2], sig_impl[3])
vm_gap_expr  = von_mises_axi(sig_gap_rr,  sig_gap_tt,  sig_gap_zz,  sig_gap_rz)

vm_field = fem.Function(W0)
vm_field.name = "von_mises_dg0"
vm_field.interpolate(fem.Expression(vm_os_expr,   W0.element.interpolation_points), cells_os)
vm_field.interpolate(fem.Expression(vm_impl_expr, W0.element.interpolation_points), cells_impl)
vm_field.interpolate(fem.Expression(vm_gap_expr,  W0.element.interpolation_points), cells_gap)

from dolfinx.fem.petsc import LinearProblem

w_test  = ufl.TestFunction(V_scal)
w_trial = ufl.TrialFunction(V_scal)
a_proj = ufl.inner(w_trial, w_test) * r * ufl.dx
L_proj = ufl.inner(vm_field, w_test) * r * ufl.dx

proj_problem = LinearProblem(
    a_proj, L_proj,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="vm_proj"
)
vm_p1_field = proj_problem.solve()
vm_p1_field.name = "von_mises"

vm_min_dg0 = float(vm_field.x.array.min())
vm_min_p1  = float(vm_p1_field.x.array.min())
print(f"[Vérification Von Mises] min DG0 (brut, calcul physique) : {vm_min_dg0:.6e} MPa")
print(f"[Vérification Von Mises] min P1  (projeté, visualisation) : {vm_min_p1:.6e} MPa")
if vm_min_dg0 < -1e-8:
    print("  -> ALERTE : le champ DG0 brut est négatif.")
else:
    print("  -> Le champ DG0 brut est bien >= 0 : le calcul physique des contraintes est correct.")

n_cells_total_diag = domain.topology.index_map(tdim).size_local
cell_nodes_diag = entities_to_geometry(domain, tdim,
                                        np.arange(n_cells_total_diag, dtype=np.int32), False)
n_geom_nodes_diag = domain.geometry.x.shape[0]
vm_env_min = np.full(n_geom_nodes_diag, np.inf)
vm_env_max = np.full(n_geom_nodes_diag, -np.inf)
vm_per_cell_diag = vm_field.x.array[:n_cells_total_diag]
nodes_per_cell_diag = cell_nodes_diag.shape[1]
flat_nodes_diag = cell_nodes_diag.flatten()
flat_vals_diag  = np.repeat(vm_per_cell_diag, nodes_per_cell_diag)
np.minimum.at(vm_env_min, flat_nodes_diag, flat_vals_diag)
np.maximum.at(vm_env_max, flat_nodes_diag, flat_vals_diag)

vm_p1_vals = vm_p1_field.x.array
overshoot_below = np.maximum(vm_env_min - vm_p1_vals, 0.0)
overshoot_above = np.maximum(vm_p1_vals - vm_env_max, 0.0)
overshoot = np.maximum(overshoot_below, overshoot_above)
n_flagged = int((overshoot > 1e-6).sum())
max_overshoot = float(overshoot.max())
worst_node = int(np.argmax(overshoot))
worst_pos = coords[worst_node]

print(f"[Vérification Von Mises] nœuds hors enveloppe DG0 voisine : "
      f"{n_flagged} / {n_geom_nodes_diag}")
print(f"[Vérification Von Mises] débordement max (au-delà du min/max voisin) : "
      f"{max_overshoot:.4f} MPa, au nœud r={worst_pos[0]:.3f} mm, z={worst_pos[1]:.3f} mm")

print(f"Von Mises Os      : min={vm_field.x.array[cells_os].min():.4f}  max={vm_field.x.array[cells_os].max():.4f}  MPa")
print(f"Von Mises Implant : min={vm_field.x.array[cells_impl].min():.4f}  max={vm_field.x.array[cells_impl].max():.4f}  MPa")
print(f"Von Mises Gap     : min={vm_field.x.array[cells_gap].min():.4f}  max={vm_field.x.array[cells_gap].max():.4f}  MPa")
print(f"u_r (modèle)      : min={u_r_field.x.array.min():.4f}  max={u_r_field.x.array.max():.4f}  mm")
print(f"u_z (modèle)      : min={u_z_field.x.array.min():.4f}  max={u_z_field.x.array.max():.4f}  mm")

with XDMFFile(MPI.COMM_WORLD,
              os.path.join(save_dir, "resultats_sans_regularisation.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    u.name = "u"
    xdmf.write_function(u)
    xdmf.write_function(u_warp_field)
    xdmf.write_function(vm_p1_field)
    xdmf.write_function(u_r_field)
    xdmf.write_function(u_z_field)
    J_field.name = "J"
    xdmf.write_function(J_field)
    xdmf.write_function(vm_field)

print("Fichier XDMF avancé sauvegardé : resultats_sans_regularisation.xdmf")

import pyvista as pv
import matplotlib.pyplot as plt
from dolfinx.plot import vtk_mesh

top_vtk, ct_vtk, geo_vtk = vtk_mesh(domain, domain.topology.dim)
grid = pv.UnstructuredGrid(top_vtk, ct_vtk, geo_vtk)
grid.cell_data["domaine"]    = cell_tags.values.astype(float)
grid.cell_data["J"]          = J_field.x.array
grid.cell_data["von_mises"]  = vm_field.x.array

u_vals = np.zeros((geo_vtk.shape[0], 3))
u_vals[:,:2] = u_array
grid.point_data["displacement"] = u_vals
grid.point_data["u_z"]          = u_array[:,1]

grid_def = grid.copy()
grid_def = grid_def.warp_by_vector("displacement", factor=warp_factor)
grid_def.cell_data["domaine"]   = cell_tags.values.astype(float)
grid_def.cell_data["J"]         = J_field.x.array
grid_def.cell_data["von_mises"] = vm_field.x.array
grid_def.point_data["u_z"]      = u_array[:,1]

vm_p95 = float(np.percentile(vm_field.x.array, 95))
grid_def.cell_data["von_mises_clip95"] = np.clip(vm_field.x.array, 0.0, vm_p95)
print(f"Von Mises — écrêtage visuel (95e percentile) : {vm_p95:.4f} MPa "
      f"(vs max réel {vm_field.x.array.max():.4f} MPa)")

CMAP = ["#D4915A", "#90C878", "#5A88D4"]
BG   = "#1a1a2e"

# VUE 1a : Initial
p1a = pv.Plotter(window_size=[700, 800])
p1a.background_color = "white"
p1a.add_text("Initial", font_size=12, color="white", position="upper_edge")
p1a.add_mesh(grid, scalars="domaine", cmap=CMAP,
             show_edges=True, edge_color="#555555", line_width=0.4,
             show_scalar_bar=False)
p1a.view_xy()
p1a.show(screenshot=os.path.join(save_dir, "vue1a_initial.png"))

# VUE 1b : Déformé
p1b = pv.Plotter(window_size=[700, 800])
p1b.background_color = "white"
p1b.add_text(f"Déformé ×{warp_factor}", font_size=12,
             color="black", position="upper_edge")
p1b.add_mesh(grid_def, scalars="domaine", cmap=CMAP,
             show_edges=True, edge_color="#555555", line_width=0.4,
             show_scalar_bar=False)
p1b.view_xy()
p1b.show(screenshot=os.path.join(save_dir, "vue1b_deforme.png"))

grid_def.save(os.path.join(save_dir, "resultats_deformes_von_mises.vtu"))
print("Maillage déformé (Von Mises exact) sauvegardé : resultats_deformes_von_mises.vtu")

# VUE 2
grid_gap_def = grid_def.extract_cells(
    np.where(cell_tags.values == TAG_GAP)[0])
grid_jmax    = grid_def.extract_cells(np.array([idx_jmax_gap]))

p2 = pv.Plotter(window_size=[700, 900])
p2.background_color = BG
p2.add_text("Déplacement u_z", font_size=12,
            color="white", position="upper_edge")
p2.add_mesh(grid_def, scalars="u_z", cmap="RdYlBu_r",
            show_edges=True, edge_color="#333333", line_width=0.3,
            smooth_shading=True,
            scalar_bar_args={"title":"u_z (mm)", "vertical":True,
                             "color":"white", "title_font_size":14,
                             "label_font_size":12})
p2.add_mesh(grid_gap_def, color="#FFD700", opacity=1.0,
            show_edges=True, edge_color="#FFA500", line_width=1.5)
p2.add_mesh(grid_jmax, color="#FF0000", opacity=1.0,
            show_edges=True, edge_color="#FF0000", line_width=2.0)
p2.view_xy()
p2.show(screenshot=os.path.join(save_dir, "vue2_uz.png"))

# VUE 2bis
p2b = pv.Plotter(window_size=[700, 900])
p2b.background_color = BG
p2b.add_text("Contrainte de Von Mises (déformé)", font_size=12,
             color="white", position="upper_edge")
p2b.add_mesh(grid_def, scalars="von_mises", cmap="turbo",
             show_edges=True, edge_color="#333333", line_width=0.3,
             smooth_shading=False,
             scalar_bar_args={"title":"Von Mises (MPa)", "vertical":True,
                              "color":"white", "title_font_size":14,
                              "label_font_size":12})
p2b.add_mesh(grid_gap_def, color="#FFD700", style="wireframe",
             line_width=2.0, opacity=1.0)
p2b.add_mesh(grid_jmax, color="#FF0000", opacity=1.0,
             show_edges=True, edge_color="#FF0000", line_width=2.0)
p2b.view_xy()
p2b.show(screenshot=os.path.join(save_dir, "vue2bis_von_mises.png"))

# VUE 3
p3 = pv.Plotter(window_size=[900, 600])
p3.background_color = BG
p3.add_text("Zoom — zone de contact", font_size=12,
            color="white", position="upper_edge")
p3.add_mesh(grid_def, scalars="u_z", cmap="RdYlBu_r",
            show_edges=True, edge_color="#333333", line_width=0.4,
            smooth_shading=True,
            scalar_bar_args={"title":"u_z (mm)", "vertical":True,
                             "color":"white"})
p3.add_mesh(grid_gap_def, color="#FFD700", opacity=1.0,
            show_edges=True, edge_color="#FFA500", line_width=1.5)
p3.add_mesh(grid_jmax, color="#FF0000", opacity=1.0,
            show_edges=True, edge_color="#FF0000", line_width=2.0)
p3.view_xy()
p3.camera.position    = (10.0, 13.5, 40.0)
p3.camera.focal_point = (10.0, 13.5, 0.0)
p3.camera.view_angle  = 30.0
p3.show(screenshot=os.path.join(save_dir, "vue3_zoom_contact.png"))

# VUE 4a
if len(hist_steps) > 0:
    hist_steps   = np.array(hist_steps)
    hist_j_min   = np.array(hist_j_min)
    hist_j_max   = np.array(hist_j_max)
    hist_uz_min  = np.array(hist_uz_min)
    hist_uz_impl = np.array(hist_uz_impl)
    hist_uz_top  = np.array(hist_uz_top)
    hist_step_num = np.array(hist_step_num)
    hist_u_plot  = np.abs(hist_steps)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(hist_u_plot, hist_j_max, color="#FF6B6B", linewidth=2.0,
             marker="o", markersize=4, label="J$_{max}$")
    ax1.plot(hist_u_plot, hist_j_min, color="#4ECDC4", linewidth=2.0,
             marker="s", markersize=4, label="J$_{min}$")
    ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8,
                alpha=0.5, label="J = 1 (référence)")
    ax1.set_xlabel("Déplacement imposé (mm)", color="black", fontsize=12)
    ax1.set_xlim(0, abs(u_imposed))
    ax1.set_ylabel("Jacobien J", color="black", fontsize=12)
    ax1.set_title("Évolution du Jacobien dans le troisième milieu", color="black", fontsize=13)
    ax1.tick_params(colors="black")
    ax1.spines[:].set_color("#444444")
    ax1.legend(facecolor="#0f0f23", labelcolor="white", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "jacobien_gap.png"), dpi=150,
                facecolor=fig1.get_facecolor())
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(hist_u_plot, np.abs(hist_uz_min), color="#FFD700", linewidth=2.0,
             marker="o", markersize=4, label="u$_z$ min global")
    ax2.plot(hist_u_plot, np.abs(hist_uz_impl), color="#FF6B6B", linewidth=2.0,
             marker="s", markersize=4, label="u$_z$ moyen implant")
    ax2.plot(hist_u_plot, hist_u_plot, color="black", linestyle="--",
             linewidth=0.8, alpha=0.5, label="u imposé (référence)")
    ax2.set_xlabel("Déplacement imposé (mm)", color="black", fontsize=12)
    ax2.set_xlim(0, abs(u_imposed))
    ax2.set_ylabel("|u$_z$| (mm)", color="white", fontsize=12)
    ax2.set_title("Descente de l'implant", color="black", fontsize=13)
    ax2.tick_params(colors="black")
    ax2.spines[:].set_color("#444444")
    ax2.legend(facecolor="#0f0f23", labelcolor="white", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "descente_implant.png"), dpi=150,
                facecolor=fig2.get_facecolor())
    plt.show()

    fig2, ax5 = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor("#1a1a2e")
    ax5.set_facecolor("#0f0f23")
    ax5.plot(hist_step_num, hist_uz_top, color="#4ECDC4",
             linewidth=2.0, marker="^", markersize=5,
             label="u$_z$ point haut implant (axe)")
    ax5.set_xlabel("Numéro de pas convergé (itération)", color="white", fontsize=12)
    ax5.set_ylabel("u$_z$ (mm)", color="white", fontsize=12)
    ax5.set_title("Descente du point haut de l'implant vs itérations",
                  color="white", fontsize=13)
    ax5.tick_params(colors="white")
    ax5.spines[:].set_color("#444444")
    ax5.grid(True, alpha=0.2, color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "descente_point_haut_implant.png"), dpi=150,
                facecolor=fig2.get_facecolor())
    plt.show()

print("Fin !")

fin = time.perf_counter()
temps = fin - debut
minutes = int(temps // 60)
secondes = int(temps % 60)
millisecondes = int((temps % 1) * 1000)
print(f"\nTemps total d'exécution : {minutes} min {secondes} s {millisecondes} ms")