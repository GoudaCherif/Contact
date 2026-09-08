"""
Maillage Os / Gap / Implant — SANS congés — implant = cavité translatée
==========================================================================
Version "coins vifs" de mesh_translation.xdmf : même principe (l'implant
est exactement la cavité de l'os, translatée verticalement de +gap0), mais
sans les congés F1/F2 — coins P4 et P5 (et leurs équivalents implant) en
angle vif, comme les tout premiers maillages de ce projet.

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

# --- Paramètres géométriques (même échelle que les maillages précédents) 
H1    = 33.6
h1    = 22.0
R1    = 20.0
r1    = 6.8639
alpha = 1.25
gap0  = 3.8     # gap = translation verticale exacte (= écart à l'axe du tout premier maillage)
lcC   = 2.0     # maille grossière
lcF   = 0.2     # maille fine

tan_a  = np.tan(alpha)
r1_top = r1 + h1 / tan_a
z5     = H1 - h1

# Points de la cavité OS (coins vifs, pas de congé)
P4 = np.array([r1_top, H1])
P5 = np.array([r1, z5])

# IMPLANT = cavité OS translatée de +gap0
shift = np.array([0.0, gap0])
i_top_axis    = np.array([0.0, H1 + gap0])
i_P4          = P4 + shift
i_P5          = P5 + shift
i_bottom_axis = np.array([0.0, z5 + gap0])

if rank == 0:
    print("=" * 62)
    print("  MAILLAGE OS-GAP-IMPLANT — SANS congés — implant = cavité translatée")
    print("=" * 62)
    print(f"  Os      : R1={R1} mm, H1={H1} mm, cavité r1={r1} -> r1_top={r1_top:.4f}")
    print(f"  gap0    = {gap0} mm (translation exacte)")
    print(f"  Implant : hauteur = h1 = {h1} mm (fixée par la cavité)")
    print("=" * 62)

# --- Gmsh 
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("os_implant_translation_sans_conges")

def Pt(pt2d, lc): return gmsh.model.occ.addPoint(pt2d[0], pt2d[1], 0, lc)
def Ln(a, b):     return gmsh.model.occ.addLine(a, b)

# --- Points OS 
p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0, lcC)
p2 = gmsh.model.occ.addPoint(R1,  0.0, 0, lcC)
p3 = gmsh.model.occ.addPoint(R1,  H1,  0, lcC)
p4 = Pt(P4, lcF)
p5 = Pt(P5, lcF)
p6 = gmsh.model.occ.addPoint(0.0, z5,  0, lcF)

# --- Points IMPLANT 
p_i_top_axis    = Pt(i_top_axis,    lcC)
p_i_P4          = Pt(i_P4,          lcF)
p_i_P5          = Pt(i_P5,          lcF)
p_i_bottom_axis = Pt(i_bottom_axis, lcF)

# --- Lignes OS 
l_bot      = Ln(p1, p2)   # bas os (encastrement)
l_ext      = Ln(p2, p3)   # extérieur os (libre)
l_top1     = Ln(p3, p4)   # épaulement haut os (libre)
l_cone_os  = Ln(p4, p5)   # cône os (interface os/gap)
l_horiz_os = Ln(p5, p6)   # horizontale os (interface os/gap)
l_axe_os   = Ln(p6, p1)   # axe os (symétrie)

cl_os = gmsh.model.occ.addCurveLoop([l_bot, l_ext, l_top1, l_cone_os, l_horiz_os, l_axe_os])
sf_os = gmsh.model.occ.addPlaneSurface([cl_os])

# --- Lignes IMPLANT (même forme que la cavité OS, translatée) 
l_impl_top   = Ln(p_i_top_axis, p_i_P4)          # haut implant (chargement)
l_cone_impl  = Ln(p_i_P4, p_i_P5)                # cône implant (interface impl/gap)
l_horiz_impl = Ln(p_i_P5, p_i_bottom_axis)        # horizontale implant (interface impl/gap)
l_axe_impl   = Ln(p_i_bottom_axis, p_i_top_axis)  # axe implant (symétrie)

cl_impl = gmsh.model.occ.addCurveLoop([l_impl_top, l_cone_impl, l_horiz_impl, l_axe_impl])
sf_impl = gmsh.model.occ.addPlaneSurface([cl_impl])

# --- GAP 
l_axe_gap = Ln(p6, p_i_bottom_axis)   # axe du gap
l_ext_gap = Ln(p_i_P4, p4)            # bord extérieur du gap (vertical, r=r1_top)

cl_gap = gmsh.model.occ.addCurveLoop([
    l_axe_gap,
    -l_horiz_impl, -l_cone_impl,   # implant, sens inverse
    l_ext_gap,
    l_cone_os, l_horiz_os          # os, sens direct
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
gmsh.model.addPhysicalGroup(1, [l_cone_os, l_horiz_os], TAG_INT_OS, name="Gamma1_up")
gmsh.model.addPhysicalGroup(1, [l_impl_top], TAG_TOP, name="Gamma2_up")
gmsh.model.addPhysicalGroup(1, [l_cone_impl, l_horiz_impl], TAG_INT_IMPL, name="Gamma2_down")
gmsh.model.addPhysicalGroup(1, [l_axe_impl], TAG_AXE_IMPL, name="Gamma2_sym")

# --- Raffinement 
contact = [l_cone_os, l_horiz_os, l_cone_impl, l_horiz_impl, l_axe_gap, l_ext_gap]
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
p1_plot.add_title("Os/Gap/Implant — sans congés, translation", font_size=11)
p1_plot.show(screenshot=os.path.join(save_dir, "geometrie_translation_sans_conges.png"))

# --- Sauvegarde XDMF 
with XDMFFile(comm, os.path.join(save_dir, "mesh_translation_sans_conges.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_tags, domain.geometry)
    xdmf.write_meshtags(facet_tags, domain.geometry)

print("Mesh sauvegardé : mesh_translation_sans_conges.xdmf")