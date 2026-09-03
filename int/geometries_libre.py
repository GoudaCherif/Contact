"""
Maillage Os / Gap / Implant — implant = cavité de l'os translatée (gap0)
==========================================================================
Reprend l'échelle géométrique des maillages coniques précédents (R1, H1,
h1, r1, alpha, Rf identiques à mesh_cones_fillets.xdmf), mais applique le
principe de la géométrie "encoche" : l'implant est EXACTEMENT la forme de
la cavité de l'os (congés F1/F2 inclus), translatée verticalement de
+gap0. Si l'implant "tombait" de gap0, son interface viendrait coïncider
exactement avec celle de l'os — le troisième milieu comble alors très
précisément cet espace, sans reste ni recouvrement.

Différence avec mesh_cones_fillets.xdmf : l'épaulement haut de l'os
(P3->P4, r1_top à R1) redevient un bord LIBRE (comme le segment C-D de la
géométrie "encoche" fournie), et non plus une interface avec le gap — la
cavité réelle (interface os/gap) commence à P4 (r1_top), pas à P3 (R1).

Tags physiques — MÊME SCHÉMA que les autres maillages du projet :
  TAG_OS=1, TAG_IMPL=2, TAG_GAP=3 (domaines)
  TAG_AXE_OS=1, TAG_EXT=2, TAG_BOT=3, TAG_INT_OS=4,
  TAG_TOP=5, TAG_INT_IMPL=6, TAG_AXE_IMPL=7 (facettes)
"""

import numpy as np
from mpi4py import MPI
import gmsh, os
from dolfinx.io import gmsh as gmshio, XDMFFile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# --- Tags physiques (identiques aux autres maillages du projet) 
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

# --- Paramètres géométriques (échelle des maillages coniques précédents) 
H1    = 33.6
h1    = 22.0
R1    = 20.0
r1    = 6.8639
alpha = 1.25
Rf    = 1.0     # rayon des congés F1, F2 (os) — repris à l'identique pour F1', F2' (implant)
gap0  = 3.8     # gap = translation verticale exacte (= écart à l'axe du tout premier maillage)
lcC   = 2.0     # maille grossière
lcF   = 0.2     # maille fine

tan_a  = np.tan(alpha)
r1_top = r1 + h1 / tan_a
z5     = H1 - h1

dr = r1_top - r1
dz = H1 - z5
Lc = np.sqrt(dr**2 + dz**2)
t_cone  = np.array([dr, dz]) / Lc
n_right = np.array([ dz, -dr]) / Lc
n_left  = np.array([-dz,  dr]) / Lc

# ---- Congés OS : F1 (haut, convexe), F2 (bas, concave) 
P4 = np.array([r1_top, H1]); P4off = P4 + Rf * n_right
t1 = (P4off[1] - (H1 - Rf)) / t_cone[1]; C1 = P4off - t1 * t_cone
T1a = np.array([C1[0], H1]); T1b = C1 - Rf * n_right

P5 = np.array([r1, z5]); P5off = P5 + Rf * n_left
t2 = (P5off[1] - (z5 + Rf)) / t_cone[1]; C2 = P5off - t2 * t_cone
T2a = C2 - Rf * n_left; T2b = np.array([C2[0], z5])

# ---- IMPLANT = cavité OS (T1a,T1b,T2a,T2b + centres C1,C2) translatée de +gap0 
shift = np.array([0.0, gap0])
i_top_axis    = np.array([0.0, H1 + gap0])
i_T1a         = T1a + shift
i_T1b         = T1b + shift
i_T2a         = T2a + shift
i_T2b         = T2b + shift
i_bottom_axis = np.array([0.0, z5 + gap0])
C1_i = C1 + shift
C2_i = C2 + shift

if rank == 0:
    print("=" * 62)
    print("  MAILLAGE OS-GAP-IMPLANT — implant = cavité translatée (gap0)")
    print("=" * 62)
    print(f"  Os      : R1={R1} mm, H1={H1} mm, cavité r1={r1} -> r1_top={r1_top:.4f}")
    print(f"  gap0    = {gap0} mm (translation exacte)")
    print(f"  Implant : hauteur = h1 = {h1} mm (fixée par la cavité, indép. de gap0)")
    print(f"  Rf      = {Rf} mm (congés F1,F2 et leurs équivalents F1',F2')")
    print("=" * 62)

# --- Gmsh 
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("os_implant_translation")

def Pt(pt2d, lc): return gmsh.model.occ.addPoint(pt2d[0], pt2d[1], 0, lc)
def Ln(a, b):     return gmsh.model.occ.addLine(a, b)
def Arc(pa, C, pb):
    pc = gmsh.model.occ.addPoint(C[0], C[1], 0, lcF)
    return gmsh.model.occ.addCircleArc(pa, pc, pb)

# --- Points OS 
p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0, lcC)   # origine (axe, bas os)
p2 = gmsh.model.occ.addPoint(R1,  0.0, 0, lcC)   # bas extérieur os
p3 = gmsh.model.occ.addPoint(R1,  H1,  0, lcC)   # haut extérieur os
p6 = Pt(np.array([0.0, z5]), lcF)                # axe bas cavité os

pT1a = Pt(T1a, lcF); pT1b = Pt(T1b, lcF)
pT2a = Pt(T2a, lcF); pT2b = Pt(T2b, lcF)

# --- Points IMPLANT 
p_i_top_axis    = Pt(i_top_axis,    lcC)
p_i_T1a         = Pt(i_T1a,         lcF)
p_i_T1b         = Pt(i_T1b,         lcF)
p_i_T2a         = Pt(i_T2a,         lcF)
p_i_T2b         = Pt(i_T2b,         lcF)
p_i_bottom_axis = Pt(i_bottom_axis, lcF)

# --- Lignes OS 
l_bot      = Ln(p1, p2)          # bas os (encastrement)
l_ext      = Ln(p2, p3)          # extérieur os (libre)
l_top1     = Ln(p3, pT1a)        # épaulement haut os (LIBRE, comme C-D de l'encoche)
arc1_os    = Arc(pT1a, C1, pT1b) # congé F1
l_cone_os  = Ln(pT1b, pT2a)      # cône os (interface os/gap)
arc2_os    = Arc(pT2a, C2, pT2b) # congé F2
l_horiz_os = Ln(pT2b, p6)        # horizontale os (interface os/gap)
l_axe_os   = Ln(p6, p1)          # axe os (symétrie)

cl_os = gmsh.model.occ.addCurveLoop([
    l_bot, l_ext, l_top1, arc1_os,
    l_cone_os, arc2_os, l_horiz_os, l_axe_os])
sf_os = gmsh.model.occ.addPlaneSurface([cl_os])

# --- Lignes IMPLANT (même forme que la cavité OS, translatée) 
l_impl_top       = Ln(p_i_top_axis, p_i_T1a)         # haut implant (chargement)
arc1_impl        = Arc(p_i_T1a, C1_i, p_i_T1b)       # congé F1'
l_cone_impl      = Ln(p_i_T1b, p_i_T2a)              # cône implant (interface impl/gap)
arc2_impl        = Arc(p_i_T2a, C2_i, p_i_T2b)       # congé F2'
l_horiz_impl     = Ln(p_i_T2b, p_i_bottom_axis)      # horizontale implant (interface impl/gap)
l_axe_impl       = Ln(p_i_bottom_axis, p_i_top_axis) # axe implant (symétrie)

cl_impl = gmsh.model.occ.addCurveLoop([
    l_impl_top, arc1_impl, l_cone_impl,
    arc2_impl, l_horiz_impl, l_axe_impl])
sf_impl = gmsh.model.occ.addPlaneSurface([cl_impl])

# --- GAP : entre la cavité OS et l'interface IMPLANT (chemin inverse), 
# fermé par l'axe (p6 -> i_bottom_axis) et le bord extérieur (i_T1a -> pT1a) 
l_axe_gap = Ln(p6, p_i_bottom_axis)     # axe du gap (nouveau)
l_ext_gap = Ln(p_i_T1a, pT1a)           # bord extérieur du gap (nouveau, vertical, r=r1_top)

cl_gap = gmsh.model.occ.addCurveLoop([
    l_axe_gap,
    -l_horiz_impl, -arc2_impl, -l_cone_impl, -arc1_impl,   # implant, sens inverse
    l_ext_gap,
    arc1_os, l_cone_os, arc2_os, l_horiz_os                # os, sens direct
])
sf_gap = gmsh.model.occ.addPlaneSurface([cl_gap])

gmsh.model.occ.synchronize()

# --- Groupes physiques 
gmsh.model.addPhysicalGroup(2, [sf_os],   TAG_OS,   name="Omega1")
gmsh.model.addPhysicalGroup(2, [sf_impl], TAG_IMPL, name="Omega2")
gmsh.model.addPhysicalGroup(2, [sf_gap],  TAG_GAP,  name="OmegaM")

gmsh.model.addPhysicalGroup(1, [l_axe_os, l_axe_gap], TAG_AXE_OS, name="Gamma1_sym")
gmsh.model.addPhysicalGroup(1, [l_ext, l_top1],       TAG_EXT,    name="Gamma1_free")
gmsh.model.addPhysicalGroup(1, [l_bot],                TAG_BOT,    name="Gamma1_down")
gmsh.model.addPhysicalGroup(1, [arc1_os, l_cone_os, arc2_os, l_horiz_os],
                             TAG_INT_OS, name="Gamma1_up")
gmsh.model.addPhysicalGroup(1, [l_impl_top], TAG_TOP, name="Gamma2_up")
gmsh.model.addPhysicalGroup(1, [arc1_impl, l_cone_impl, arc2_impl, l_horiz_impl],
                             TAG_INT_IMPL, name="Gamma2_down")
gmsh.model.addPhysicalGroup(1, [l_axe_impl], TAG_AXE_IMPL, name="Gamma2_sym")

# --- Raffinement 
contact = [arc1_os, l_cone_os, arc2_os, l_horiz_os,
           arc1_impl, l_cone_impl, arc2_impl, l_horiz_impl,
           l_axe_gap, l_ext_gap]
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", contact)
gmsh.model.mesh.field.setNumber(1, "Sampling", 300)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", lcF)
gmsh.model.mesh.field.setNumber(2, "SizeMax", lcC)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(2, "DistMax", 5.0)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.model.mesh.setTransfiniteCurve(l_cone_os,   80)
gmsh.model.mesh.setTransfiniteCurve(l_cone_impl, 80)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")

# --- Conversion DOLFINx 
mesh_data  = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
domain     = mesh_data.mesh
cell_tags  = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
gmsh.finalize()

# --- Infos 
if rank == 0:
    nc = domain.topology.index_map(2).size_global
    ns = domain.topology.index_map(0).size_global
    print(f"Mesh — Cellules : {nc} | Sommets : {ns}")
    coords = domain.geometry.x
    print(f"  r : [{coords[:,0].min():.3f}, {coords[:,0].max():.3f}] mm")
    print(f"  z : [{coords[:,1].min():.3f}, {coords[:,1].max():.3f}] mm")
    for tag in np.unique(cell_tags.values):
        name = {1: "Os", 2: "Implant", 3: "Gap"}.get(int(tag), "???")
        n = (cell_tags.values == tag).sum()
        print(f"  Tag {tag} ({name}) : {n} cellules")
    print(f"  Tags facettes : {np.unique(facet_tags.values)}")

# --- Visualisation 
import pyvista as pv
from dolfinx.plot import vtk_mesh

top_vtk, ct_vtk, geo_vtk = vtk_mesh(domain, domain.topology.dim)
grid = pv.UnstructuredGrid(top_vtk, ct_vtk, geo_vtk)
grid.cell_data["domaine"] = cell_tags.values.astype(float)

save_dir = os.path.dirname(os.path.abspath(__file__))

p1_plot = pv.Plotter(window_size=[700, 900])
p1_plot.background_color = "white"
p1_plot.add_mesh(grid, scalars="domaine",
                  cmap=["#D4915A", "#90C878", "#5A88D4"],
                  show_edges=True, categories=True,
                  edge_color="#555555", line_width=0.6,
                  scalar_bar_args={"title": "1=Os | 2=Implant | 3=Gap",
                                   "vertical": True, "color": "black"})
p1_plot.view_xy()
p1_plot.camera.zoom(1.1)
p1_plot.add_title("Os/Gap/Implant — implant = cavité translatée", font_size=11)
p1_plot.show(screenshot=os.path.join(save_dir, "geometrie_translation.png"))

# --- Sauvegarde XDMF 
with XDMFFile(comm, os.path.join(save_dir, "mesh_translation.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_tags, domain.geometry)
    xdmf.write_meshtags(facet_tags, domain.geometry)

print("Mesh sauvegardé : mesh_translation.xdmf")