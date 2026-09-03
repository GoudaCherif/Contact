"""
Maillage Os / Gap / Implant — AVEC congés (Rf = 1 mm) aux interfaces
======================================================================
Reprend la topologie à 3 domaines (Os, Implant, Gap = "troisième milieu")
du maillage original, mais remplace 3 des 4 coins vifs de l'interface
os/gap et gap/implant par des congés circulaires — même construction
géométrique (tangentes, centres de cercle) que le maillage 2-domaines
"os_implant_4fillets".

Congés (Rf = 1 mm) :
  F1 : haut os / cône os              (convexe ext)   — ex-coin P4
  F2 : cône os / horizontale os       (concave int)   — ex-coin P5
  F4 : horizontale impl / cône impl   (convexe impl)  — ex-coin P9

PAS de congé en P6 (jonction axe os / horizontale os) — angle droit conservé.
PAS de congé en P8 (jonction axe impl / horizontale impl) — angle droit conservé.
PAS de congé en P10 (ex-F5, cône impl / haut impl) — angle droit conservé.
  Raison : le bord extérieur libre du gap (nouveau, absent du maillage à
  2 domaines de référence) part de ce même point vers P3. Le point tangent
  amont du congé F5 se trouve, pour cette géométrie précise, à ~0.05 mm de
  cette droite — largement inférieur au rayon de congé (1 mm) — ce qui
  provoque une auto-intersection de la frontière du Gap (Gmsh : "Unable to
  recover the edge... on surface 3"). Ce conflit est spécifique à l'ajout
  du troisième milieu ; il n'existe pas dans le maillage à 2 domaines de
  référence, qui ne comporte pas ce bord extérieur.

Tags physiques — IDENTIQUES au maillage original (mesh_cones.xdmf), pour
que le script de simulation existant tourne SANS AUCUNE MODIFICATION sur
ce nouveau maillage :
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

# --- Tags physiques (identiques au maillage original) 
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

# --- Paramètres géométriques 
# r1 et alpha repris de la géométrie ORIGINALE (troisième milieu, avec une
# vraie épaisseur de gap) — PAS du script à congés fourni, dont les valeurs
# (r1=6.8039, alpha=1.2502181) sont réglées pour un CONTACT quasi-direct
# (Nitsche, sans gap) : la distance perpendiculaire entre les deux cônes y
# est de seulement 0.003 mm, contre 0.31 mm avec les valeurs ci-dessous —
# largement en dessous de la taille de maille (lcF=0.2 mm), ce qui rend le
# gap localement quasi inexistant et fait exploser numériquement le calcul
# (J jusqu'à 200+, Von Mises jusqu'à plusieurs milliers de MPa).
H1    = 33.6
h1    = 22.0
R1    = 20.0
r1    = 6.8639
alpha = 1.25
r2    = 7.8
H2    = 25.6
H12   = 41.0
Rf    = 1.0     # rayon des congés
lcC   = 2.0     # maille grossière
lcF   = 0.2     # maille fine

tan_a  = np.tan(alpha)
r1_top = r1 + h1 / tan_a
r2_top = r2 + H2 / tan_a
z5 = H1 - h1
z8 = H12 - H2

dr = r1_top - r1
dz = H1 - z5
Lc = np.sqrt(dr**2 + dz**2)
t_cone  = np.array([dr, dz]) / Lc
n_right = np.array([ dz, -dr]) / Lc
n_left  = np.array([-dz,  dr]) / Lc

# ---- Congés : mêmes calculs (tangentes, centres) que le maillage 2-domaines 
# F1 : ex-P4, convexe ext (haut os / cône os)
P4 = np.array([r1_top, H1]); P4off = P4 + Rf * n_right
t1 = (P4off[1] - (H1 - Rf)) / t_cone[1]; C1 = P4off - t1 * t_cone
T1a = np.array([C1[0], H1]); T1b = C1 - Rf * n_right

# F2 : ex-P5, concave int (cône os / horizontale os), T2b forcé à z=z5
P5 = np.array([r1, z5]); P5off = P5 + Rf * n_left
t2 = (P5off[1] - (z5 + Rf)) / t_cone[1]; C2 = P5off - t2 * t_cone
T2a = C2 - Rf * n_left; T2b = np.array([C2[0], z5])

# F4 : ex-P9, convexe impl (horizontale impl / cône impl)
P9 = np.array([r2, z8]); P9off = P9 + Rf * n_left
t4 = (z8 + Rf - P9off[1]) / t_cone[1]; C4 = P9off + t4 * t_cone
T4a = np.array([C4[0], z8]); T4b = C4 - Rf * n_left

# F5 : SUPPRIMÉ — angle droit conservé en P10 (voir docstring en tête de fichier)
P10 = np.array([r2_top, H12])

if rank == 0:
    sep = "=" * 62
    print(sep)
    print("  MAILLAGE OS-GAP-IMPLANT — congés Rf = 1 mm aux interfaces")
    print(sep)
    print(f"  Os      : R={R1} mm, H={H1} mm")
    print(f"  Gap     : de z={z5:.2f} à z={z8:.2f} mm (axe)")
    print(f"  Implant : r={r2} mm, H={H2} mm")
    print(f"  Rf      = {Rf} mm  (rayon des congés F1,F2,F4 — F5 en angle droit)")
    print(sep)

# --- Gmsh 
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("cones_gap_fillets")

def Pt(r, z, lc): return gmsh.model.occ.addPoint(r, z, 0, lc)
def Ln(a, b):     return gmsh.model.occ.addLine(a, b)
def Arc(pa, cx, cz, pb):
    pc = gmsh.model.occ.addPoint(cx, cz, 0, lcF)
    return gmsh.model.occ.addCircleArc(pa, pc, pb)

# Points fixes
p1 = Pt(0.0, 0.0, lcC)   # origine (axe, bas os)
p2 = Pt(R1,  0.0, lcC)   # bas extérieur os
p3 = Pt(R1,  H1,  lcC)   # haut extérieur os
p6 = Pt(0.0, z5,  lcF)   # coin axe/horiz os — angle droit conservé
p7 = Pt(0.0, H12, lcC)   # haut axe implant
p8 = Pt(0.0, z8,  lcF)   # coin axe/horiz impl — angle droit conservé
p10 = Pt(r2_top, H12, lcF)  # coin droit (ex-F5), haut du cône implant

# Points tangents des congés
pT1a = Pt(*T1a, lcF); pT1b = Pt(*T1b, lcF)
pT2a = Pt(*T2a, lcF); pT2b = Pt(*T2b, lcF)
pT4a = Pt(*T4a, lcF); pT4b = Pt(*T4b, lcF)

# --- Contour OS 
l_bot      = Ln(p1, p2)            # bas os (encastrement)
l_ext      = Ln(p2, p3)            # extérieur os (libre)
l_top1     = Ln(p3, pT1a)          # haut os → tangent F1 (interface os/gap)
arc1       = Arc(pT1a, *C1, pT1b)  # congé F1
l_cone_os  = Ln(pT1b, pT2a)        # cône os (interface os/gap)
arc2       = Arc(pT2a, *C2, pT2b)  # congé F2
l_horiz_os = Ln(pT2b, p6)          # horizontale os (interface os/gap)
l_axe_os   = Ln(p6, p1)            # axe os (symétrie)

cl_os = gmsh.model.occ.addCurveLoop([
    l_bot, l_ext, l_top1, arc1,
    l_cone_os, arc2, l_horiz_os, l_axe_os])
sf_os = gmsh.model.occ.addPlaneSurface([cl_os])

# --- Contour IMPLANT 
l_axe_impl   = Ln(p7, p8)           # axe implant (symétrie)
l_horiz_impl = Ln(p8, pT4a)         # horizontale impl (interface impl/gap)
arc4         = Arc(pT4a, *C4, pT4b) # congé F4
l_cone_impl  = Ln(pT4b, p10)        # cône impl (interface impl/gap) — jusqu'au coin droit
l_top_impl   = Ln(p10, p7)          # haut impl (chargement) — depuis le coin droit

cl_impl = gmsh.model.occ.addCurveLoop([
    l_axe_impl, l_horiz_impl, arc4,
    l_cone_impl, l_top_impl])
sf_impl = gmsh.model.occ.addPlaneSurface([cl_impl])

# --- Contour GAP (troisième milieu) 
# Réutilise les MÊMES courbes que les interfaces os/gap et impl/gap
# ci-dessus (partagées entre les deux surfaces adjacentes -> maillage
# conforme à l'interface), plus deux courbes propres au gap : l'axe
# (p6 -> p8) et le bord extérieur libre (p10 -> p3), analogues à
# L11 et L12 dans le maillage original sans congés.
l_axe_gap = Ln(p6, p8)   # axe du gap (nouveau, propre au gap)
l_ext_gap = Ln(p10, p3)  # bord extérieur libre du gap (nouveau, propre au gap)

cl_gap = gmsh.model.occ.addCurveLoop([
    l_axe_gap,
    l_horiz_impl, arc4, l_cone_impl,             # partagé avec IMPLANT
    l_ext_gap,
    l_top1, arc1, l_cone_os, arc2, l_horiz_os    # partagé avec OS
])
sf_gap = gmsh.model.occ.addPlaneSurface([cl_gap])

gmsh.model.occ.synchronize()

# --- Groupes physiques (mêmes tags que le maillage original) 
gmsh.model.addPhysicalGroup(2, [sf_os],   TAG_OS,   name="Omega1")
gmsh.model.addPhysicalGroup(2, [sf_impl], TAG_IMPL, name="Omega2")
gmsh.model.addPhysicalGroup(2, [sf_gap],  TAG_GAP,  name="OmegaM")

gmsh.model.addPhysicalGroup(1, [l_axe_os],  TAG_AXE_OS, name="Gamma1_sym")
gmsh.model.addPhysicalGroup(1, [l_ext],     TAG_EXT,    name="Gamma1_free")
gmsh.model.addPhysicalGroup(1, [l_bot],     TAG_BOT,    name="Gamma1_down")
gmsh.model.addPhysicalGroup(1, [l_top1, arc1, l_cone_os, arc2, l_horiz_os],
                             TAG_INT_OS, name="Gamma1_up")
gmsh.model.addPhysicalGroup(1, [l_top_impl], TAG_TOP,      name="Gamma2_up")
gmsh.model.addPhysicalGroup(1, [l_horiz_impl, arc4, l_cone_impl],
                             TAG_INT_IMPL, name="Gamma2_down")
gmsh.model.addPhysicalGroup(1, [l_axe_impl], TAG_AXE_IMPL, name="Gamma2_sym")

# --- Raffinement 
contact = [l_top1, arc1, l_cone_os, arc2, l_horiz_os,
           l_horiz_impl, arc4, l_cone_impl]
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
p1_plot.add_title("Os/Gap/Implant — congés Rf=1mm", font_size=12)
p1_plot.show(screenshot=os.path.join(save_dir, "geometrie_fillets.png"))

# --- Sauvegarde XDMF 
with XDMFFile(comm, os.path.join(save_dir, "mesh_cones_fillets.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_tags, domain.geometry)
    xdmf.write_meshtags(facet_tags, domain.geometry)

print("Mesh sauvegardé : mesh_cones_fillets.xdmf")