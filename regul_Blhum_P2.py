import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile
import ufl
from petsc4py import PETSc
import os

# ── Tags physiques ──────────────────────────────────────────────────────────
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

# ── Paramètres ─────────────────────────────────────────────────────────────
E_r       = 822.0
E_z       = 1352.0
nu_rtheta = 0.3
nu_rz     = 0.3
G_rz      = 399.0
nu_zr     = nu_rz * E_z / E_r
E_2       = 113000.0
nu_2      = 0.3
gamma     = 1e-6
mu_m      = 1.0
alpha_r   = 1.0           # <-- réduit de 1000 à 1
u_imposed = -1.7
n_steps   = 20            # <-- augmenté (pas initial plus petit)
warp_factor = 2.0

# ── Espace P2 ──────────────────────────────────────────────────────────────
V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
u = fem.Function(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
r = x[0]
print(f"DOFs : {V.dofmap.index_map.size_global * 2}")

# ── Cinématique et fonctions d'énergie (inchangées) ──────────────────────
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

def W_lineaire(u, r, C):
    eps = strain_axi(u, r)
    return 0.5 * ufl.dot(eps, C * eps)

def W_neohookean_iso(u, r, mu, gamma):
    F  = deformation_gradient_axi(u, r)
    C  = F.T * F
    Ic = ufl.tr(C)
    J  = ufl.det(F)
    return gamma * (mu / 2) * (J**(-2/3) * Ic - 3)

def W_regularisation_Bluhm(u, r, gamma, alpha_r):
    F     = deformation_gradient_axi(u, r)
    gradF = ufl.grad(F)
    return (gamma / 2) * alpha_r * ufl.inner(gradF, gradF)

dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

W_total = (
    W_lineaire(u, r, C_os)                        * r * dx(TAG_OS)   +
    W_lineaire(u, r, C_impl)                      * r * dx(TAG_IMPL) +
    W_neohookean_iso(u, r, mu_m, gamma)           * r * dx(TAG_GAP)  +
    W_regularisation_Bluhm(u, r, gamma, alpha_r)  * r * dx(TAG_GAP)
)

F_form = ufl.derivative(W_total, u, v)
J_form = ufl.derivative(F_form, u, ufl.TrialFunction(V))
print("Formes variationnelles définies (régularisation Bluhm).")

# ── Conditions aux limites ─────────────────────────────────────────────────
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

from dolfinx.mesh import entities_to_geometry
top_nodes = np.unique(entities_to_geometry(domain, 1, top_facets, False).flatten())
print(f"z noeuds TOP : min={domain.geometry.x[top_nodes,1].min():.3f}  "
      f"max={domain.geometry.x[top_nodes,1].max():.3f}")

# ── Solveur avec options MUMPS renforcées ────────────────────────────────
from dolfinx.fem.petsc import NonlinearProblem

"""petsc_opts = {
    "snes_type"                 : "newtonls",
    "snes_linesearch_type"      : "bt",
    "snes_atol"                 : 1e-8,
    "snes_rtol"                 : 1e-8,
    "snes_max_it"               : 500,
    "snes_monitor"              : None,
    "ksp_type"                  : "preonly",
    "pc_type"                   : "lu",
    "pc_factor_mat_solver_type" : "mumps",
    # Options MUMPS pour tolérer des pivots plus petits et éviter INFOG(1)=-3
    "mat_mumps_cntl_1"          : 0.01,    # seuil de pivotement (par défaut 0.01)
    "mat_mumps_icntl_14"        : 200,     # mémoire
    "mat_mumps_icntl_24"        : 1,       # détection de pivot nul
}"""
# En cas d'échec persistant, décommenter les lignes ci‑dessous pour passer à GMRES+HYPRE
# mais cela nécessite que HYPRE soit installé.

petsc_opts = {
    "snes_type"                 : "newtonls",
    "snes_linesearch_type"      : "bt",
    "snes_atol"                 : 1e-8,
    "snes_rtol"                 : 1e-8,
    "snes_max_it"               : 500,
    "snes_monitor"              : None,
    "ksp_type"                  : "gmres",
    "pc_type"                   : "hypre",
    "pc_hypre_type"             : "boomeramg",
    "ksp_atol"                  : 1e-10,
    "ksp_rtol"                  : 1e-10,
    "ksp_max_it"                : 1000,
}

problem = NonlinearProblem(
    F_form, u, bcs=bcs, J=J_form,
    petsc_options=petsc_opts,
    petsc_options_prefix="tmc"
)

# ── Post‑traitement ────────────────────────────────────────────────────────
W0 = fem.functionspace(domain, ("DG", 0))
J_expr = fem.Expression(ufl.det(deformation_gradient_axi(u, r)),
                         W0.element.interpolation_points)
J_field = fem.Function(W0)
cells_gap  = np.where(cell_tags.values == TAG_GAP)[0]
cells_os   = np.where(cell_tags.values == TAG_OS)[0]
cells_impl = np.where(cell_tags.values == TAG_IMPL)[0]

V_p1 = fem.functionspace(domain, ("Lagrange", 1, (2,)))
u_p1 = fem.Function(V_p1)

cells_impl_nodes = np.unique(
    entities_to_geometry(domain, 2, cells_impl, False).flatten()
)

hist_steps     = []
hist_j_min     = []
hist_j_max     = []
hist_uz_min    = []
hist_uz_impl   = []
snapshots      = []

# ── Résolution avec pas adaptatif ──────────────────────────────────────────
u_per_step = u_imposed / n_steps
current_u = 0.0
step_count = 0
max_attempts = 300
pas_min = 1e-6 * abs(u_imposed)

print(f"Résolution — pas initial de {u_per_step:.4f} mm")

while abs(current_u) < abs(u_imposed) and step_count < max_attempts:
    target_u = current_u + u_per_step
    if abs(target_u) > abs(u_imposed):
        target_u = u_imposed
        u_per_step = target_u - current_u

    u_imp.value = target_u
    print(f"Tentative {step_count+1} : u={target_u:.4f} mm (incr={u_per_step:.4f})")

    try:
        converged = problem.solve()
        reason = problem.solver.getConvergedReason()
        if reason > 0:
            current_u = target_u
            step_count += 1
            J_field.interpolate(J_expr)
            j_min = J_field.x.array[cells_gap].min()
            j_max = J_field.x.array[cells_gap].max()
            u_arr = u.x.array.reshape(-1, 2)
            uz_min = u_arr[:,1].min()
            u_p1.interpolate(u)
            u_arr_p1 = u_p1.x.array.reshape(-1, 2)
            uz_impl = u_arr_p1[cells_impl_nodes, 1].mean()
            print(f"  OK (reason={reason}) Step {step_count} | u={current_u:.3f} | "
                  f"u_z_min={uz_min:.4f} | u_z_impl={uz_impl:.4f} | J=[{j_min:.4f},{j_max:.4f}]")
            hist_steps.append(current_u)
            hist_j_min.append(j_min)
            hist_j_max.append(j_max)
            hist_uz_min.append(uz_min)
            hist_uz_impl.append(uz_impl)
            # Optionnel : réaugmenter le pas progressivement
            # u_per_step = min(u_per_step * 1.2, abs(u_imposed - current_u)/2)
        else:
            print(f"  Échec (reason={reason}), réduction du pas")
            u_per_step /= 2.0
            if abs(u_per_step) < pas_min:
                print("  Pas trop petit, arrêt.")
                break
    except Exception as e:
        print(f"  Exception : {e}, réduction du pas")
        u_per_step /= 2.0
        if abs(u_per_step) < pas_min:
            print("  Pas trop petit, arrêt.")
            break

print(f"Résolution terminée après {step_count} pas convergés.")

# ── Post‑traitement avancé (distorsion) ───────────────────────────────────
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
print(f"Distorsion (√(3 I2_dev)) : min={dist_gap.min():.4e}  max={dist_gap.max():.4e}  mean={dist_gap.mean():.4e}")

# ── Résumé final ───────────────────────────────────────────────────────────
u_p1.interpolate(u)
u_array = u_p1.x.array.reshape(-1, 2)
eps_zz  = fem.Function(W0)
eps_zz.interpolate(fem.Expression(u[1].dx(1), W0.element.interpolation_points))
J_field.interpolate(J_expr)

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

# ── Sauvegarde XDMF ──────────────────────────────────────────────────────
with XDMFFile(MPI.COMM_WORLD,
              os.path.join(save_dir, "resultats_Bluhm_robuste.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    J_field.name = "J"
    xdmf.write_function(J_field)
    u_p1.name = "u"
    xdmf.write_function(u_p1)
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

print("Fichier XDMF avancé sauvegardé : resultats_Bluhm_robuste.xdmf")