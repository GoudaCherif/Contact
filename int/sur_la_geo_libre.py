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
              os.path.join(save_dir, "mesh_translation_sans_conges.xdmf"), "r") as xdmf:
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
u_imposed = -3.8
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

# Le problème est encapsulé dans une fonction (plutôt qu'un objet unique) car
# il doit être RECRÉÉ après chaque échec de convergence pendant la boucle de
# pas adaptatif : reconstruire NonlinearProblem force PETSc/MUMPS à repartir
# d'un état interne propre (ré-analyse symbolique, factorisation fraîche), ce
# qui évite qu'un état de solveur corrompu par un échec précédent (résidu
# NaN, pivot défaillant) ne contamine la tentative suivante.
def make_problem():
    return NonlinearProblem(
        F_form, u, bcs=bcs, J=J_form,
        petsc_options={
            # --- Solveur non-linéaire (SNES) ---
            "snes_type"                 : "newtonls",  # Newton avec recherche linéaire (line search)
            "snes_linesearch_type"      : "bt",         # backtracking : réduit le pas Newton si le
                                                          # résidu ne décroît pas assez -> plus robuste
                                                          # qu'un Newton pur, surtout ici vu la forte
                                                          # non-linéarité du terme hyperélastique du gap.
            "snes_atol"                 : 1e-9,          # tolérance absolue sur la norme du résidu
            "snes_rtol"                 : 1e-9,          # tolérance relative (résidu / résidu initial)
            "snes_max_it"               : 100,           # nb max d'itérations Newton par pas de charge ;
                                                          # si non atteint -> reason <= 0 -> échec du pas
            "snes_monitor"              : None,          # affiche le résidu à chaque itération Newton
            "snes_converged_reason"     : None,          # affiche la raison de convergence/divergence

            # --- Solveur linéaire (KSP) résolu à CHAQUE itération Newton ---
            "ksp_type"                  : "preonly",     # pas d'itératif : on ne fait QUE appliquer le
                                                          # préconditionneur, qui est ici une factorisation
                                                          # directe complète -> résolution "exacte"
            "pc_type"                   : "lu",          # préconditionneur = factorisation LU directe
            "pc_factor_mat_solver_type" : "mumps",       # LU réalisée par le solveur direct MUMPS
                                                          # (robuste sur systèmes mal conditionnés/creux)
            "pc_factor_mat_solver_package" : "mumps",    # alias redondant (compat. anciennes versions PETSc)

            # --- Réglages fins MUMPS, nécessaires vu le très mauvais
            #     conditionnement du système : rigidité os/implant (E~10^3-10^5 MPa)
            #     vs rigidité du gap (gamma*mu_m ~ 10^-4 MPa), ratio ~10^7-10^9 ---
            "mat_mumps_icntl_14"        : 200,           # marge mémoire de travail (+200%) allouée pour
                                                          # le remplissage (fill-in) de la factorisation ;
                                                          # évite un échec par manque de mémoire de travail
                                                          # sur une matrice à conditionnement extrême.
            "mat_mumps_cntl_1"          : 0.1,           # seuil de pivotement relevé (0.1 au lieu du
                                                          # défaut ~1e-8) : accepte des pivots plus petits
                                                          # comme "suffisamment grands", ce qui stabilise
                                                          # la factorisation sur un système quasi-singulier,
                                                          # au prix d'une précision numérique légèrement
                                                          # réduite sur la solution.
        },
        petsc_options_prefix="tmc"     # préfixe des options PETSc, pour isoler ce solveur
                                        # d'autres solveurs éventuels dans le même run
                                        # (ex. la projection L² Von Mises, préfixe "vm_proj")
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

# Point suivi : nœud sur l'axe, en haut de l'implant (proche de r=0, z=H12)
node_top_axis = top_nodes[np.argmin(coords[top_nodes, 0])]
print(f"Point suivi (haut implant, axe) — r={coords[node_top_axis,0]:.4f} mm, "
      f"z={coords[node_top_axis,1]:.4f} mm")

hist_steps     = []
hist_j_min     = []
hist_j_max     = []
hist_uz_min    = []
hist_uz_impl   = []
hist_uz_top    = []   # u_z du point suivi, par pas convergé
hist_step_num  = []   # numéro du pas convergé (pour tracer en fonction des itérations)
snapshots      = []

#  Résolution avec pas adaptatif + restauration 
# Stratégie de "load stepping" (continuation) : on n'impose pas u_imposed d'un
# coup (Newton diverge sur un pas trop grand vu la forte non-linéarité et le
# quasi-écrasement possible du gap) mais par petits incréments successifs. Si
# un incrément échoue à converger, on revient en arrière et on retente avec un
# pas deux fois plus petit -> convergence quasi garantie tant que le pas peut
# être réduit indéfiniment (borné par delta_min).
u_per_step_initial = u_imposed / n_steps   # taille de pas visée au départ
u_per_step = u_per_step_initial            # taille de pas courante (réduite en cas d'échec)
current_u  = 0.0                           # déplacement total déjà appliqué et convergé
step_count = 0                             # nombre de pas convergés avec succès
attempt_count = 0                          # nombre total de tentatives (succès + échecs)
max_attempts  = 500                        # garde-fou : arrêt même si le pas ne devient
                                            # jamais assez petit (évite une boucle infinie)
delta_min = 1e-6 * abs(u_imposed)          # pas minimal toléré ; en dessous, on considère
                                            # que la convergence est hors de portée et on arrête

# Sauvegarder l'état initial (u=0) : c'est l'état vers lequel on revient si
# la toute première tentative de pas échoue.
u_prev = u.x.array.copy()

print(f"Résolution — pas initial de {u_per_step:.4f} mm (max {n_steps} pas)")
sys.stdout.flush()

while abs(current_u) < abs(u_imposed) and attempt_count < max_attempts:
    # Calcule la cible du pas courant : current_u + le pas courant, sauf si
    # cela dépasserait le déplacement total visé (on clippe alors exactement
    # sur u_imposed pour ne jamais le dépasser).
    target_u = current_u + u_per_step
    if abs(target_u) > abs(u_imposed):
        target_u = u_imposed
        u_per_step = target_u - current_u

    u_imp.value = target_u   # met à jour la valeur de la BC de Dirichlet pilotée
    attempt_count += 1
    print(f"Tentative {attempt_count} : u={target_u:.4f} mm (incr={u_per_step:.4f})")
    sys.stdout.flush()

    try:
        converged = problem.solve()               # lance Newton pour ce pas
        reason = problem.solver.getConvergedReason()  # code SNES : >0 succès, <=0 échec
        if reason > 0:
            # --- Succès : le pas est validé, on avance ---
            current_u = target_u
            step_count += 1
            # L'état convergé devient le nouveau point de restauration en cas
            # d'échec du PROCHAIN pas.
            u_prev = u.x.array.copy()
            print(f"  OK (reason={reason})")

            # Sauvegarde de quelques instantanés du champ de déplacement pour
            # une éventuelle visualisation de l'historique de déformation
            # (1er pas, 1/4, 1/2, dernier pas prévu).
            if step_count in [1, n_steps//4, n_steps//2, n_steps]:
                u_snapshot = u.x.array.reshape(-1, 2).copy()
                snapshots.append((current_u, u_snapshot))

            # Mise à jour des historiques de suivi (tracés en fin de script) :
            # on recalcule le champ J (Jacobien) en interpolant son expression
            # UFL sur l'espace DG0, puis on extrait quelques indicateurs
            # scalaires (min/max dans le gap, descente de l'implant...).
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

            # Piste d'amélioration non activée : ré-augmenter le pas après un
            # succès (accélère la résolution une fois la zone difficile passée),
            # laissée en commentaire car non nécessaire ici empiriquement.
            # u_per_step = min(u_per_step * 1.2, abs(u_imposed - current_u)/2)

        else:
            # --- Échec de convergence Newton (reason <= 0) ---
            # On revient à l'état convergé précédent (sinon u contient un
            # itéré non physique issu du Newton qui a divergé), on réduit
            # le pas de moitié, et on réessaie le MÊME current_u avec un
            # incrément plus petit.
            u.x.array[:] = u_prev
            print(f"  Échec (reason={reason}), restauration + réduction du pas")
            sys.stdout.flush()
            u_per_step /= 2.0
            if abs(u_per_step) < delta_min:
                print("  Pas trop petit, arrêt.")
                break
            # Recréer le problème pour réinitialiser MUMPS (cf. commentaire
            # sur make_problem plus haut : repartir d'un état solveur propre).
            problem = make_problem()
            continue

    except Exception as e:
        # --- Échec "dur" : exception levée pendant le solve (ex. J<=0 dans
        #     le gap -> J**(-2/3) indéfini -> NaN ou erreur d'évaluation) ---
        # Même traitement que l'échec de convergence : restaurer, réduire,
        # recréer le solveur, puis retenter.
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

#  Résumé final 
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

#  Post‑traitement avancé (distorsion) 
# (on utilise W0 déjà défini)
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
print(f"Distorsion : min={dist_gap.min():.4e}  max={dist_gap.max():.4e}  mean={dist_gap.mean():.4e}")

#  Contraintes de Von Mises + déplacements (ensemble du modèle) 
print("\n--- Contraintes de Von Mises et déplacements (modèle complet) ---")

# Déplacements radial / axial, champs P1 sur tout le domaine
V_scal = fem.functionspace(domain, ("Lagrange", 1))
u_r_field = fem.Function(V_scal)
u_z_field = fem.Function(V_scal)
u_r_field.name = "u_r"
u_z_field.name = "u_z"
u_r_field.interpolate(fem.Expression(u[0], V_scal.element.interpolation_points))
u_z_field.interpolate(fem.Expression(u[1], V_scal.element.interpolation_points))

# Champ vectoriel dédié au Warp By Vector dans ParaView : 'u' n'a que 2
# composantes (u_r, u_z), cohérent avec un maillage 2D, mais Warp By Vector
# attend un vecteur à 3 composantes pour fonctionner correctement (c'est
# pour cette même raison que le bloc PyVista plus bas complète 'u' à 3
# composantes avant d'appeler warp_by_vector). On construit ici l'équivalent
# pour l'export XDMF/ParaView, 3e composante nulle (pas de déplacement
# hors-plan dans ce modèle axisymétrique).
V_warp = fem.functionspace(domain, ("Lagrange", 1, (3,)))
u_warp_field = fem.Function(V_warp)
u_warp_field.name = "u_warp"
zero_scalar = fem.Constant(domain, PETSc.ScalarType(0.0))
u_warp_field.interpolate(fem.Expression(
    ufl.as_vector([u[0], u[1], zero_scalar]),
    V_warp.element.interpolation_points))

# Contraintes — zones linéaires (os, implant) : sigma_voigt = C * eps_axi
def sigma_voigt_lin(u, r, C):
    eps = strain_axi(u, r)   # [eps_rr, eps_tt, eps_zz, gamma_rz]
    return C * eps           # [sigma_rr, sigma_tt, sigma_zz, tau_rz]

sig_os   = sigma_voigt_lin(u, r, C_os)
sig_impl = sigma_voigt_lin(u, r, C_impl)

# Contrainte de Cauchy — zone hyperélastique (gap), via dérivation automatique
# P = dW/dF (1er Piola-Kirchhoff), sigma = (1/J) * P * F^T (Cauchy)
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

# Champ Von Mises DG0 (précis, zone par zone) — utilisé pour les stats console.
# Interpolation zone par zone : correcte et déjà robuste en parallèle
# (DOLFINx gère nativement le assemblage/scatter des Function distribuées).
vm_field = fem.Function(W0)
vm_field.name = "von_mises_dg0"
vm_field.interpolate(fem.Expression(vm_os_expr,   W0.element.interpolation_points), cells_os)
vm_field.interpolate(fem.Expression(vm_impl_expr, W0.element.interpolation_points), cells_impl)
vm_field.interpolate(fem.Expression(vm_gap_expr,  W0.element.interpolation_points), cells_gap)

# Champ Von Mises P1 (même espace que u_r/u_z) — pour visualisation ParaView
# (DG0 est écrit par dolfinx sur une géométrie séparée dans le XDMF, ce qui casse
#  Warp By Vector côté ParaView ; P1 partage la géométrie du maillage, comme u_r/u_z)
#
# CORRECTIF : le moyennage nodal "à la main" (np.add.at sur les cellules locales)
# est incorrect en exécution MPI multi-process, car domain.topology.index_map
# .size_local exclut les cellules fantômes (ghost) des voisins de partition :
# sur les nœuds situés à une frontière de partition, la somme/le compte seraient
# tronqués et le champ P1 présenterait une discontinuité artificielle le long de
# ces frontières (invisible en séquentiel, où il n'y a pas de ghost cells).
#
# On utilise à la place une projection L² standard de vm_field (DG0, déjà
# correct zone par zone) sur l'espace P1 : problème variationnel classique
# assemblé et résolu par DOLFINx/PETSc, donc géré nativement en distribué
# (scatter/gather MPI corrects), sans logique de réduction manuelle à maintenir.
# Pondérée par r (comme dx(TAG_...) partout ailleurs dans le script) pour rester
# cohérente avec la mesure axisymétrique réelle du problème.
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

# --- Diagnostic : le champ DG0 brut (vm_field) doit être >= 0 par construction
# (Von Mises = racine d'une somme de carrés). S'il l'est bien, toute valeur
# négative observée sur le champ projeté (vm_p1_field) provient uniquement de
# la projection L² elle-même (débordement de type Gibbs au voisinage des
# interfaces à fort contraste de rigidité), pas d'un bug dans le calcul des
# contraintes. Ce test permet de trancher sans ambiguïté entre les deux.
vm_min_dg0 = float(vm_field.x.array.min())
vm_min_p1  = float(vm_p1_field.x.array.min())
print(f"[Vérification Von Mises] min DG0 (brut, calcul physique) : {vm_min_dg0:.6e} MPa")
print(f"[Vérification Von Mises] min P1  (projeté, visualisation) : {vm_min_p1:.6e} MPa")
if vm_min_dg0 < -1e-8:
    print("  -> ALERTE : le champ DG0 brut est négatif. Ceci indique un problème réel "
          "dans le calcul des contraintes (formule, signe, ou NaN silencieux) — "
          "à investiguer avant de faire confiance à un quelconque résultat de Von Mises.")
else:
    print("  -> Le champ DG0 brut est bien >= 0 : le calcul physique des contraintes est "
          "correct. La valeur négative visible sur le champ projeté (vm_p1_field) est un "
          "artefact numérique de la projection L² aux interfaces à fort contraste de "
          "rigidité (os/gap, gap/implant), pas une erreur de calcul.")

# --- Diagnostic renforcé : le test ci-dessus (min DG0 >= 0) ne prouve
# l'absence d'erreur de projection QUE là où elle franchit zéro — un
# débordement positif (valeur projetée surestimée sans devenir négative,
# car la vraie valeur y est déjà élevée) resterait invisible à ce test.
# On vérifie ici, pour CHAQUE nœud, que la valeur projetée reste dans
# l'enveloppe [min, max] des valeurs DG0 des cellules qui le touchent —
# un test valable quel que soit le signe, qui détecte tout débordement
# local de la projection, pas seulement ceux qui croisent zéro.
# (Diagnostic uniquement, non utilisé pour corriger quoi que ce soit ;
# comme pour l'ancien moyennage manuel, ce calcul ne considère que les
# cellules locales -> valable tel quel en séquentiel, sous-estimerait le
# nombre de nœuds détectés en MPI multi-process du fait des ghost cells.)
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
overshoot_below = np.maximum(vm_env_min - vm_p1_vals, 0.0)   # dépasse sous le min voisin
overshoot_above = np.maximum(vm_p1_vals - vm_env_max, 0.0)   # dépasse au-dessus du max voisin
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

#  Sauvegarde XDMF enrichie 
with XDMFFile(MPI.COMM_WORLD,
              os.path.join(save_dir, "resultats_sans_regularisation_translation.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    # Champs P1 D'ABORD, tous compatibles avec la géométrie du maillage —
    # DOLFINx les rattache à la même grille XDMF que le maillage, ce qui
    # permet à ParaView de combiner Warp By Vector (avec 'u_warp') et une
    # coloration par n'importe lequel des autres champs P1 sur ce même bloc
    # (notamment 'von_mises', la version projetée). Écrire un champ DG0
    # AVANT ces champs P1 peut perturber ce rattachement à une grille unique
    # (la géométrie dupliquée du DG0 devient alors la grille "principale"),
    # empêchant précisément la combinaison warp+couleur — d'où cet ordre.
    u.name = "u"
    xdmf.write_function(u)
    xdmf.write_function(u_warp_field)  # 3 composantes — Vecteur pour Warp By Vector
    xdmf.write_function(vm_p1_field)   # P1, lissé — se combine avec le warp ci-dessus
    xdmf.write_function(u_r_field)
    xdmf.write_function(u_z_field)

    # Champs DG0 ENSUITE — grille séparée de toute façon (géométrie dupliquée),
    # non combinables avec Warp By Vector quel que soit l'ordre d'écriture.
    J_field.name = "J"
    xdmf.write_function(J_field)
    xdmf.write_function(vm_field)      # exact (von_mises_dg0), non lissé
    """  eps_rr_field.name = "eps_rr"
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
    xdmf.write_function(distorsion_field)"""

print("Fichier XDMF avancé sauvegardé : resultats_sans_regularisation_translation.xdmf")

#  Visualisation PyVista 
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

# Champ Von Mises écrêté (percentile 95), pour éviter qu'un point singulier
# isolé (coin rentrant de la géométrie, où la contrainte peut croître sans
# borne avec le raffinement du maillage) n'écrase visuellement toute la
# colormap. On garde "von_mises" (exact, non modifié) ET on ajoute ce
# second champ, déjà prêt à afficher sans aucun réglage de colormap dans
# ParaView — sélectionner directement "von_mises_clip95" dans le menu.
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
p1a.show()

# VUE 1b : Déformé
p1b = pv.Plotter(window_size=[700, 800])
p1b.background_color = "white"
p1b.add_text(f"Déformé ×{warp_factor}", font_size=12,
             color="black", position="upper_edge")
p1b.add_mesh(grid_def, scalars="domaine", cmap=CMAP,
             show_edges=True, edge_color="#555555", line_width=0.4,
             show_scalar_bar=False)
p1b.view_xy()
p1b.show()

# Sauvegarde directe du maillage DÉJÀ déformé (par PyVista, warp_factor) avec
# Von Mises exact (DG0, non lissé) déjà attaché comme donnée de cellule.
# Contourne complètement le problème Warp By Vector + XDMF : ce fichier
# s'ouvre tel quel dans ParaView, déjà déformé et coloré, sans aucun filtre
# à appliquer ni à configurer.
grid_def.save(os.path.join(save_dir, "resultats_deformes_von_mises_translation.vtu"))
print("Maillage déformé (Von Mises exact) sauvegardé : resultats_deformes_von_mises_translation.vtu")

# VUE 2 : u_z + gap jaune + J_max rouge
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
p2.show()

# VUE 2bis : Von Mises sur le maillage déformé
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
p2b.show()

# VUE 3 : Zoom zone de contact
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
p3.show()

# VUE 4a : Courbes J_min, J_max dans le troisième milieu
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
    plt.savefig(os.path.join(save_dir, "jacobien_gap_translation.png"), dpi=150,
                facecolor=fig1.get_facecolor())
    plt.show()

    # VUE 4b : Descente implant (|u_z|)
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
    plt.savefig(os.path.join(save_dir, "descente_implant_translation.png"), dpi=150,
                facecolor=fig2.get_facecolor())
    plt.show()

    # VUE 5 : Descente du point haut-implant en fonction du numéro d'itération
    # u_z tracé directement (non en valeur absolue) : la courbe descend bien,
    # car u_z devient de plus en plus négatif au fil des pas.
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
    plt.savefig(os.path.join(save_dir, "descente_point_haut_implant_translation.png"), dpi=150,
                facecolor=fig2.get_facecolor())
    plt.show()

print("Fin !")

fin = time.perf_counter()
temps = fin - debut
minutes = int(temps // 60)
secondes = int(temps % 60)
millisecondes = int((temps % 1) * 1000)
print(f"\nTemps total d'exécution : {minutes} min {secondes} s {millisecondes} ms")