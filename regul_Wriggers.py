import time
debut = time.perf_counter()

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
import ufl
from petsc4py import PETSc
from basix.ufl import element, mixed_element
import os
import sys
#import pyvista as pv
#import matplotlib.pyplot as plt
from dolfinx.plot import vtk_mesh
from dolfinx.mesh import entities_to_geometry

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
gamma     = 1e-5          # rigidité du gap
mu_m      = 1.0
beta_1    = 1e4           # pénalité pour p (rotation)
beta_2    = 10.0          # pénalité pour q (Jacobien) – peut être 0
alpha_r   = 100.0         # régularisation sur gradients de p et q
eps_stab  = 1e-10          # stabilisation p,q hors gap (forte)
u_imposed = -1.7
n_steps   = 30
warp_factor = 2.0
kappa = 1e6

#  Espace mixte (u, p, q) en P1 
cell   = domain.basix_cell()
elem_u = element("Lagrange", cell, 1, shape=(2,))
elem_p = element("Lagrange", cell, 1)
elem_q = element("Lagrange", cell, 1)
W_space = fem.functionspace(domain, mixed_element([elem_u, elem_p, elem_q]))

w  = fem.Function(W_space)
dw = ufl.TestFunction(W_space)

# Extraction des champs (u, p, q) en tant qu'expressions UFL
u, p, q = ufl.split(w)
du, dp, dq = ufl.split(dw)

x = ufl.SpatialCoordinate(domain)
r = x[0]
print(f"DOFs total : {W_space.dofmap.index_map.size_global * W_space.dofmap.index_map_bs}")
sys.stdout.flush()

#  Cinématique 
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

def tan_phi(u):
    """Proxy de la rotation en 2D méridien — Wriggers eq. (9)"""
    F_2d = ufl.Identity(2) + ufl.grad(u)
    return (F_2d[0,1] - F_2d[1,0]) / (F_2d[0,0] + F_2d[1,1] + 1e-12)

#  Tenseurs élasticité 
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

"""def W_neohookean_iso(u, r, mu, gamma):
    F  = deformation_gradient_axi(u, r)
    C  = F.T * F
    Ic = ufl.tr(C)
    J  = ufl.det(F)
    return gamma * (mu / 2) * (J**(-2/3) * Ic - 3)"""



def W_neohookean_iso(u, r, mu, gamma, kappa):
    F = deformation_gradient_axi(u, r)
    C = F.T * F
    Ic = ufl.tr(C)
    J = ufl.det(F)
    # Partie déviatorique (distorsion)
    W_iso = (mu / 2) * (J**(-2/3) * Ic - 3)
    # Partie volumique (pénalité sur J)
    W_vol = (kappa / 2) * (ufl.ln(J))**2
    return gamma * (W_iso + W_vol)

def W_phi_reg(u, p, gamma, alpha_r, beta_1):
    tp = tan_phi(u)
    return (gamma / 2) * (beta_1 * (tp - p)**2 + alpha_r * ufl.dot(ufl.grad(p), ufl.grad(p)))

def W_J_reg(u, q, r, gamma, alpha_r, beta_2):
    F = deformation_gradient_axi(u, r)
    J = ufl.det(F)
    return (gamma / 2) * (beta_2 * (J - q)**2 + alpha_r * ufl.dot(ufl.grad(q), ufl.grad(q)))

dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

# Stabilisation p/q hors gap (pénalité de masse)
W_stab = eps_stab * (p*p + q*q) * r * (dx(TAG_OS) + dx(TAG_IMPL))

W_total = (
    W_lineaire(u, r, C_os)                     * r * dx(TAG_OS)   +
    W_lineaire(u, r, C_impl)                   * r * dx(TAG_IMPL) +
   # W_neohookean_iso(u, r, mu_m, gamma)        * r * dx(TAG_GAP)  +
   W_neohookean_iso(u, r, mu_m, gamma, kappa) * r * dx(TAG_GAP)+
    W_phi_reg(u, p, gamma, alpha_r, beta_1)    * r * dx(TAG_GAP)  +
    W_J_reg(u, q, r, gamma, alpha_r, beta_2)   * r * dx(TAG_GAP)  +
    W_stab
)

F_form = ufl.derivative(W_total, w, dw)
J_form = ufl.derivative(F_form, w, ufl.TrialFunction(W_space))
print("Formes variationnelles définies (formulation p/q).")
sys.stdout.flush()

#  Conditions aux limites 
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

bot_facets  = facet_tags.find(TAG_BOT)
top_facets  = facet_tags.find(TAG_TOP)
axe_facets  = np.concatenate([
    facet_tags.find(TAG_AXE_OS),
    facet_tags.find(TAG_AXE_IMPL)
])

V_u = W_space.sub(0)
dofs_bot_r = fem.locate_dofs_topological(V_u.sub(0), fdim, bot_facets)
dofs_bot_z = fem.locate_dofs_topological(V_u.sub(1), fdim, bot_facets)
dofs_axe_r = fem.locate_dofs_topological(V_u.sub(0), fdim, axe_facets)
dofs_top_z = fem.locate_dofs_topological(V_u.sub(1), fdim, top_facets)

zero  = fem.Constant(domain, PETSc.ScalarType(0.0))
u_imp = fem.Constant(domain, PETSc.ScalarType(0.0))

bcs = [
    fem.dirichletbc(zero,  dofs_bot_r, V_u.sub(0)),
    fem.dirichletbc(zero,  dofs_bot_z, V_u.sub(1)),
    fem.dirichletbc(zero,  dofs_axe_r, V_u.sub(0)),
    fem.dirichletbc(u_imp, dofs_top_z, V_u.sub(1)),
]
print(f"CL — BOT:{len(dofs_bot_z)} AXE:{len(dofs_axe_r)} TOP:{len(dofs_top_z)}")
sys.stdout.flush()

#  Solveur 
from dolfinx.fem.petsc import NonlinearProblem

def make_problem():
    return NonlinearProblem(
        F_form, w, bcs=bcs, J=J_form,
        petsc_options={
            "snes_type"                 : "newtonls",
            "snes_linesearch_type"      : "bt",
            "snes_atol"                 : 1e-8,
            "snes_rtol"                 : 1e-8,
            "snes_max_it"               : 200,
            "snes_monitor"              : None,
            "snes_converged_reason"     : None,
            "ksp_type"                  : "preonly",
            "pc_type"                   : "lu",
            "pc_factor_mat_solver_type" : "mumps",
            "mat_mumps_cntl_1"          : 0.1,
            "mat_mumps_icntl_14"        : 200,
            "ksp_atol" : 1e-10,
            "ksp_rtol" : 1e-10,
        },
        petsc_options_prefix="tmc_pq"
    )

problem = make_problem()

#  Post‑traitement 
W0 = fem.functionspace(domain, ("DG", 0))
J_expr = fem.Expression(ufl.det(deformation_gradient_axi(u, r)),
                         W0.element.interpolation_points)
J_field = fem.Function(W0)

# Espace P1 pour projection de u (pour les statistiques et visualisation)
V_u_out = fem.functionspace(domain, ("Lagrange", 1, (2,)))
u_p1_out = fem.Function(V_u_out)

cells_gap  = np.where(cell_tags.values == TAG_GAP)[0]
cells_os   = np.where(cell_tags.values == TAG_OS)[0]
cells_impl = np.where(cell_tags.values == TAG_IMPL)[0]
cells_impl_nodes = np.unique(
    entities_to_geometry(domain, 2, cells_impl, False).flatten()
)

hist_steps   = []
hist_j_min   = []
hist_j_max   = []
hist_uz_min  = []
hist_uz_impl = []

#  Résolution avec pas adaptatif 
u_per_step_initial = u_imposed / n_steps
u_per_step = u_per_step_initial
current_u = 0.0
step_count = 0
attempt_count = 0
max_attempts = 500
delta_min = 1e-6 * abs(u_imposed)

# Sauvegarde de l'état initial (w)
w_prev = w.x.array.copy()

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
            # Sauvegarder l'état convergé
            w_prev = w.x.array.copy()

            # Extraire u en P1 pour les statistiques
            u_expr = fem.Expression(u, V_u_out.element.interpolation_points)
            u_p1_out.interpolate(u_expr)
            u_arr = u_p1_out.x.array.reshape(-1, 2)

            # Mettre à jour J
            J_field.interpolate(J_expr)
            j_min = J_field.x.array[cells_gap].min()
            j_max = J_field.x.array[cells_gap].max()
            uz_min = u_arr[:,1].min()
            uz_impl = u_arr[cells_impl_nodes, 1].mean()

            hist_steps.append(current_u)
            hist_j_min.append(j_min)
            hist_j_max.append(j_max)
            hist_uz_min.append(uz_min)
            hist_uz_impl.append(uz_impl)

            print(f"  OK (reason={reason}) Step {step_count} | u={current_u:.3f} | "
                  f"u_z_min={uz_min:.4f} | u_z_impl={uz_impl:.4f} | J=[{j_min:.4f},{j_max:.4f}]")
            sys.stdout.flush()
        else:
            # Échec : restaurer l'état précédent
            w.x.array[:] = w_prev
            print(f"  Échec (reason={reason}), restauration + réduction du pas")
            sys.stdout.flush()
            u_per_step /= 2.0
            if abs(u_per_step) < delta_min:
                print("  Pas trop petit, arrêt.")
                break
            # Recréer le problème pour réinitialiser MUMPS
            problem = make_problem()
            continue
    except Exception as e:
        w.x.array[:] = w_prev
        print(f"  Exception : {e}, restauration + réduction du pas")
        sys.stdout.flush()
        u_per_step /= 2.0
        if abs(u_per_step) < delta_min:
            print("  Pas trop petit, arrêt.")
            break
        problem = make_problem()
        continue

print(f"Résolution terminée après {step_count} pas convergés.")
sys.stdout.flush()

#  Résumé final (avec projection P1) 
# On a déjà u_p1_out mis à jour au dernier pas convergé
u_array = u_p1_out.x.array.reshape(-1, 2)

# Calcul de eps_zz sur P1 (via la projection)
eps_zz_p1 = fem.Function(W0)
eps_zz_p1.interpolate(fem.Expression(u_p1_out[1].dx(1), W0.element.interpolation_points))
J_field.interpolate(J_expr)

# Masque pour l'axe (coords est en P1)
mask_axe = np.isclose(coords[:,0], 0.0, atol=0.01)

print(f"u_z — min:{u_array[:,1].min():.4f}  max:{u_array[:,1].max():.4f} mm")
print(f"u_r — min:{u_array[:,0].min():.4f}  max:{u_array[:,0].max():.4f} mm")
print(f"J gap  — min:{J_field.x.array[cells_gap].min():.4f}  "
      f"max:{J_field.x.array[cells_gap].max():.4f}")
print(f"ε_zz os      : {eps_zz_p1.x.array[cells_os].mean():.4e}")
print(f"ε_zz gap     : {eps_zz_p1.x.array[cells_gap].mean():.4e}")
print(f"ε_zz implant : {eps_zz_p1.x.array[cells_impl].mean():.4e}")

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

idx_jmin_gap = cells_gap[np.argmin(J_field.x.array[cells_gap])]
jmin_nodes   = entities_to_geometry(domain, 2,
                                    np.array([idx_jmin_gap]), False).flatten()
jmin_center  = coords[jmin_nodes].mean(axis=0)
print(f"Cellule J_min — centre : r={jmin_center[0]:.3f} mm, z={jmin_center[1]:.3f} mm")

#  Post‑traitement avancé : distorsion 
# On utilise les expressions UFL de u (qui est encore valide)
eps_rr_expr = u[0].dx(0)
eps_tt_expr = u[0] / r
eps_zz_expr = u[1].dx(1)
eps_rz_expr = 0.5 * (u[0].dx(1) + u[1].dx(0))

eps_rr_field = fem.Function(W0); eps_rr_field.interpolate(fem.Expression(eps_rr_expr, W0.element.interpolation_points))
eps_tt_field = fem.Function(W0); eps_tt_field.interpolate(fem.Expression(eps_tt_expr, W0.element.interpolation_points))
eps_zz_field = fem.Function(W0); eps_zz_field.interpolate(fem.Expression(eps_zz_expr, W0.element.interpolation_points))
eps_rz_field = fem.Function(W0); eps_rz_field.interpolate(fem.Expression(eps_rz_expr, W0.element.interpolation_points))

trace_eps = eps_rr_expr + eps_tt_expr + eps_zz_expr
dev_rr = eps_rr_expr - trace_eps / 3
dev_tt = eps_tt_expr - trace_eps / 3
dev_zz = eps_zz_expr - trace_eps / 3
dev_rz = eps_rz_expr
I2_dev_expr = 0.5 * (dev_rr**2 + dev_tt**2 + dev_zz**2 + 2 * dev_rz**2)
I2_dev_field = fem.Function(W0); I2_dev_field.interpolate(fem.Expression(I2_dev_expr, W0.element.interpolation_points))

distorsion_expr = ufl.sqrt(3 * I2_dev_expr)
distorsion_field = fem.Function(W0); distorsion_field.interpolate(fem.Expression(distorsion_expr, W0.element.interpolation_points))

# Statistiques
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

#  Sauvegarde XDMF enrichie 
with XDMFFile(MPI.COMM_WORLD,
              os.path.join(save_dir, "resultats_pq_avance.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    J_field.name = "J"
    xdmf.write_function(J_field)
    u_p1_out.name = "u"
    xdmf.write_function(u_p1_out)
    eps_rr_field.name = "eps_rr"
    xdmf.write_function(eps_rr_field)
    eps_tt_field.name = "eps_tt"
    xdmf.write_function(eps_tt_field)
    eps_zz_field.name = "eps_zz"
    xdmf.write_function(eps_zz_field)
    eps_rz_field.name = "eps_rz"
    xdmf.write_function(eps_rz_field)
    I2_dev_field.name = "I2_dev"
    xdmf.write_function(I2_dev_field)
    distorsion_field.name = "distorsion"
    xdmf.write_function(distorsion_field)

print("Fichier XDMF sauvegardé : resultats_pq_avance.xdmf")

#  Visualisation PyVista 
"""top_vtk, ct_vtk, geo_vtk = vtk_mesh(domain, domain.topology.dim)
grid = pv.UnstructuredGrid(top_vtk, ct_vtk, geo_vtk)
grid.cell_data["domaine"] = cell_tags.values.astype(float)
grid.cell_data["J"]       = J_field.x.array

u_vals = np.zeros((geo_vtk.shape[0], 3))
u_vals[:,:2] = u_array
grid.point_data["displacement"] = u_vals
grid.point_data["u_z"]          = u_array[:,1]

grid_def = grid.copy()
grid_def = grid_def.warp_by_vector("displacement", factor=warp_factor)
grid_def.cell_data["domaine"] = cell_tags.values.astype(float)
grid_def.cell_data["J"]       = J_field.x.array
grid_def.point_data["u_z"]    = u_array[:,1]

CMAP = ["#D4915A", "#90C878", "#5A88D4"]
BG   = "#1a1a2e"

# VUE 1 : Initial vs Déformé
p1 = pv.Plotter(shape=(1,2), window_size=[1200,800])
p1.subplot(0,0)
p1.background_color = BG
p1.add_text("Initial", font_size=12, color="white", position="upper_edge")
p1.add_mesh(grid, scalars="domaine", cmap=CMAP,
            show_edges=True, edge_color="#555555", line_width=0.4)
p1.view_xy()
p1.subplot(0,1)
p1.background_color = "white"
p1.add_text(f"Déformé ×{warp_factor} — p/q", font_size=12,
            color="black", position="upper_edge")
p1.add_mesh(grid_def, scalars="domaine", cmap=CMAP,
            show_edges=True, edge_color="#555555", line_width=0.4)
p1.view_xy()
p1.show()

# VUE 2 : u_z + gap jaune + J_max rouge
grid_gap_def = grid_def.extract_cells(np.where(cell_tags.values == TAG_GAP)[0])
grid_jmax    = grid_def.extract_cells(np.array([idx_jmax_gap]))

p2 = pv.Plotter(window_size=[700, 900])
p2.background_color = BG
p2.add_text("Déplacement u_z — p/q", font_size=12, color="white", position="upper_edge")
p2.add_mesh(grid_def, scalars="u_z", cmap="RdYlBu_r",
            show_edges=True, edge_color="#333333", line_width=0.3,
            smooth_shading=True,
            scalar_bar_args={"title":"u_z (mm)", "vertical":True,
                             "color":"white", "title_font_size":14, "label_font_size":12})
p2.add_mesh(grid_gap_def, color="#FFD700", opacity=1.0,
            show_edges=True, edge_color="#FFA500", line_width=1.5)
p2.add_mesh(grid_jmax, color="#FF0000", opacity=1.0,
            show_edges=True, edge_color="#FF0000", line_width=2.0)
p2.view_xy()
p2.show()

# VUE 3 : Zoom zone de contact
p3 = pv.Plotter(window_size=[900, 600])
p3.background_color = BG
p3.add_text("Zoom — zone de contact — p/q", font_size=12, color="white", position="upper_edge")
p3.add_mesh(grid_def, scalars="u_z", cmap="RdYlBu_r",
            show_edges=True, edge_color="#333333", line_width=0.4,
            smooth_shading=True,
            scalar_bar_args={"title":"u_z (mm)", "vertical":True, "color":"white"})
p3.add_mesh(grid_gap_def, color="#FFD700", opacity=1.0,
            show_edges=True, edge_color="#FFA500", line_width=1.5)
p3.add_mesh(grid_jmax, color="#FF0000", opacity=1.0,
            show_edges=True, edge_color="#FF0000", line_width=2.0)
p3.view_xy()
p3.camera.position    = (10.0, 13.5, 40.0)
p3.camera.focal_point = (10.0, 13.5, 0.0)
p3.camera.view_angle  = 30.0
p3.show()

# VUE 4 : Courbes J_min, J_max + descente implant
if len(hist_steps) > 0:
    hist_steps   = np.array(hist_steps)
    hist_j_min   = np.array(hist_j_min)
    hist_j_max   = np.array(hist_j_max)
    hist_uz_min  = np.array(hist_uz_min)
    hist_uz_impl = np.array(hist_uz_impl)
    hist_u_plot  = np.abs(hist_steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    ax1 = axes[0]
    ax1.set_facecolor("#0f0f23")
    ax1.plot(hist_u_plot, hist_j_max, color="#FF6B6B", linewidth=2.0,
             marker="o", markersize=4, label="J$_{max}$")
    ax1.plot(hist_u_plot, hist_j_min, color="#4ECDC4", linewidth=2.0,
             marker="s", markersize=4, label="J$_{min}$")
    ax1.axhline(y=1.0, color="white", linestyle="--", linewidth=0.8,
                alpha=0.5, label="J = 1")
    ax1.set_xlabel("Déplacement imposé (mm)", color="white", fontsize=12)
    ax1.set_xlim(0, abs(u_imposed))
    ax1.set_ylabel("Jacobien J", color="white", fontsize=12)
    ax1.set_title("Évolution du Jacobien — p/q", color="white", fontsize=13)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#444444")
    ax1.legend(facecolor="#0f0f23", labelcolor="white", fontsize=11)
    ax1.grid(True, alpha=0.2, color="white")

    ax2 = axes[1]
    ax2.set_facecolor("#0f0f23")
    ax2.plot(hist_u_plot, np.abs(hist_uz_min), color="#FFD700", linewidth=2.0,
             marker="o", markersize=4, label="u$_z$ min global")
    ax2.plot(hist_u_plot, np.abs(hist_uz_impl), color="#FF6B6B", linewidth=2.0,
             marker="s", markersize=4, label="u$_z$ moyen implant")
    ax2.plot(hist_u_plot, hist_u_plot, color="white", linestyle="--",
             linewidth=0.8, alpha=0.5, label="u imposé")
    ax2.set_xlabel("Déplacement imposé (mm)", color="white", fontsize=12)
    ax2.set_xlim(0, abs(u_imposed))
    ax2.set_ylabel("|u$_z$| (mm)", color="white", fontsize=12)
    ax2.set_title("Descente de l'implant — p/q", color="white", fontsize=13)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444444")
    ax2.legend(facecolor="#0f0f23", labelcolor="white", fontsize=11)
    ax2.grid(True, alpha=0.2, color="white")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "courbes_pq.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.show()"""

#  Fin 
fin = time.perf_counter()
temps = fin - debut
minutes = int(temps // 60)
secondes = int(temps % 60)
millisecondes = int((temps % 1) * 1000)
print(f"\nTemps total d'exécution : {minutes} min {secondes} s {millisecondes} ms")
print("Fin du script.")