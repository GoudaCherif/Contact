"""
Maillage Os / Gap / Implant 

Géométrie (mm) :
  Os     : Omega1 - tag 1
  Implant: Omega2 - tag 2
  Gap    : OmegaM - tag 3

Tags facettes :
  TAG_AXE_OS   = 1  (Gamma1_sym)   - axe os
  TAG_EXT      = 2  (Gamma1_free)  - bord extérieur os
  TAG_BOT      = 3  (Gamma1_down)  - bas os (encastrement)
  TAG_INT_OS   = 4  (Gamma1_up)    - interface os/gap
  TAG_TOP      = 5  (Gamma2_up)    - haut implant (chargement)
  TAG_INT_IMPL = 6  (Gamma2_down)  - interface implant/gap
  TAG_AXE_IMPL = 7  (Gamma2_sym)   - axe implant
"""

import numpy as np
from mpi4py import MPI
import gmsh
import os
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# --- Tags physiques 
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
H1    = 33.6
h1    = 22.0
R1    = 20.0
r1    = 6.8639
alpha = 1.25
r2    = 7.8
H2    = 25.6
H12   = 41.0

lcC = 2.0   # maille grossière
lcF = 0.1   # maille fine

# Points dérivés
r1_top = r1 + h1 / np.tan(alpha)   # = 14.17 mm
r2_top = r2 + H2 / np.tan(alpha)   # = 16.31 mm
z5     = H1 - h1                    # = 11.6 mm
z8     = H12 - H2                   # = 15.4 mm

print("Génération du maillage !")
print(f"  Os    : R={R1} mm, H={H1} mm")
print(f"  Gap   : de z={z5:.2f} à z={z8:.2f} mm (axe)")
print(f"  Implant : r={r2} mm, H={H2} mm")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("cones")

# --- Points 
# Os
P1  = gmsh.model.occ.addPoint(0.0,    0.0,  0, lcC)   # origine
P2  = gmsh.model.occ.addPoint(R1,     0.0,  0, lcC)   # bas extérieur
P3  = gmsh.model.occ.addPoint(R1,     H1,   0, lcF)   # haut extérieur
P4  = gmsh.model.occ.addPoint(r1_top, H1,   0, lcC)   # haut interface conique os
P5  = gmsh.model.occ.addPoint(r1,     z5,   0, lcF)   # bas interface conique os
P6  = gmsh.model.occ.addPoint(0.0,    z5,   0, lcF)   # axe bas gap (os)

# Implant
P7  = gmsh.model.occ.addPoint(0.0,    H12,  0, lcC)   # haut axe implant
P8  = gmsh.model.occ.addPoint(0.0,    z8,   0, lcF)   # bas axe implant
P9  = gmsh.model.occ.addPoint(r2,     z8,   0, lcF)   # bas interface conique implant
P10 = gmsh.model.occ.addPoint(r2_top, H12,  0, lcF)   # haut interface conique implant

# --- Lignes 
# Os
L1  = gmsh.model.occ.addLine(P1, P2)   # bas os
L2  = gmsh.model.occ.addLine(P2, P3)   # extérieur os
L3  = gmsh.model.occ.addLine(P3, P4)   # haut os (extérieur)
L4  = gmsh.model.occ.addLine(P4, P5)   # interface conique os/gap
L5  = gmsh.model.occ.addLine(P5, P6)   # interface horizontale os/gap
L6  = gmsh.model.occ.addLine(P6, P1)   # axe os

# Implant
L7  = gmsh.model.occ.addLine(P7,  P8)  # axe implant
L8  = gmsh.model.occ.addLine(P8,  P9)  # bas implant (horizontal)
L9  = gmsh.model.occ.addLine(P9,  P10) # interface conique implant/gap
L10 = gmsh.model.occ.addLine(P10, P7)  # haut implant

# Gap
L11 = gmsh.model.occ.addLine(P6,  P8)  # axe gap
L12 = gmsh.model.occ.addLine(P10, P3)  # extérieur gap

# --- Surfaces 
# Os : P1-P2-P3-P4-P5-P6
cl_os   = gmsh.model.occ.addCurveLoop([L1, L2, L3, L4, L5, L6])
sf_os   = gmsh.model.occ.addPlaneSurface([cl_os])

# Implant : P7-P8-P9-P10
cl_impl = gmsh.model.occ.addCurveLoop([L7, L8, L9, L10])
sf_impl = gmsh.model.occ.addPlaneSurface([cl_impl])

# Gap : entre os et implant
# Délimité par : L11(axe gap), L8(bas impl), L9(interface impl), L12(ext), L3(haut os ext), L4(interface os), L5(interface os horiz)
cl_gap  = gmsh.model.occ.addCurveLoop([L11, L8, L9, L12, L3, L4, L5])
sf_gap  = gmsh.model.occ.addPlaneSurface([cl_gap])

gmsh.model.occ.synchronize()

# --- Groupes physiques
gmsh.model.addPhysicalGroup(2, [sf_os],   TAG_OS,   name="Omega1")
gmsh.model.addPhysicalGroup(2, [sf_impl], TAG_IMPL, name="Omega2")
gmsh.model.addPhysicalGroup(2, [sf_gap],  TAG_GAP,  name="OmegaM")

gmsh.model.addPhysicalGroup(1, [L6],       TAG_AXE_OS,   name="Gamma1_sym")
gmsh.model.addPhysicalGroup(1, [L2],       TAG_EXT,      name="Gamma1_free")
gmsh.model.addPhysicalGroup(1, [L1],       TAG_BOT,      name="Gamma1_down")
gmsh.model.addPhysicalGroup(1, [L5, L4, L3], TAG_INT_OS, name="Gamma1_up")
gmsh.model.addPhysicalGroup(1, [L10],      TAG_TOP,      name="Gamma2_up")
gmsh.model.addPhysicalGroup(1, [L8, L9],   TAG_INT_IMPL, name="Gamma2_down")
gmsh.model.addPhysicalGroup(1, [L7],       TAG_AXE_IMPL, name="Gamma2_sym")

# --- Raffinement 
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [L4, L5, L8, L9])
gmsh.model.mesh.field.setNumber(1,  "Sampling",   250)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField",  1)
gmsh.model.mesh.field.setNumber(2, "SizeMin",  lcF)
gmsh.model.mesh.field.setNumber(2, "SizeMax",  lcC)
gmsh.model.mesh.field.setNumber(2, "DistMin",  0.1)
gmsh.model.mesh.field.setNumber(2, "DistMax",  5.0)

gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.model.mesh.setTransfiniteCurve(L4, 100)
gmsh.model.mesh.setTransfiniteCurve(L9, 100)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")

# --- Conversion DOLFINx 
mesh_data  = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
domain     = mesh_data.mesh
cell_tags  = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
gmsh.finalize()

# --- Infos 
nc = domain.topology.index_map(2).size_global
ns = domain.topology.index_map(0).size_global
print(f"Mesh — Cellules : {nc} | Sommets : {ns}")
coords = domain.geometry.x
print(f"  r : [{coords[:,0].min():.3f}, {coords[:,0].max():.3f}] mm")
print(f"  z : [{coords[:,1].min():.3f}, {coords[:,1].max():.3f}] mm")
for tag in np.unique(cell_tags.values):
    name = {1:"Os", 2:"Implant", 3:"Gap"}.get(tag, "???")
    print(f"  Tag {tag} ({name}) : {(cell_tags.values == tag).sum()} cellules")
print(f"  Tags facettes : {np.unique(facet_tags.values)}")

# --- Visualisation 
import pyvista as pv
from dolfinx.plot import vtk_mesh

top_vtk, ct_vtk, geo_vtk = vtk_mesh(domain, domain.topology.dim)
grid = pv.UnstructuredGrid(top_vtk, ct_vtk, geo_vtk)
grid.cell_data["domaine"] = cell_tags.values.astype(float)

BG = "#1a1a2e"

p1 = pv.Plotter(window_size=[700, 900])
p1.background_color = "white"
p1.add_mesh(grid, scalars="domaine",
            cmap=["#D4915A", "#90C878", "#5A88D4"],
            show_edges=True, categories=True,
            edge_color="#555555", line_width=0.6,
            scalar_bar_args={"title":"1=Os | 2=Implant | 3=Gap",
                             "vertical":True, "color":"white"})
p1.view_xy()
p1.camera.zoom(1.1)
p1.add_title("Géométries OS-IMPLANT", font_size=12)
p1.show()

# Zoom zone de contact
y_coords  = grid.cell_centers().points[:, 1]
mask      = (y_coords >= z5 - 2.0) & (y_coords <= z8 + 6.0)
grid_zoom = grid.extract_cells(np.where(mask)[0])

p2 = pv.Plotter(window_size=[900, 600])
p2.background_color = "white"
p2.add_mesh(grid_zoom, scalars="domaine",
            cmap=["#D4915A", "#90C878", "#5A88D4"],
            show_edges=True, edge_color="#555555", line_width=0.8)
p2.view_xy()
p2.camera.zoom(1.5)
p2.add_title("Zoom — Zone de contact", font_size=11)
#p2.show()

# --- Sauvegarde XDMF 
save_dir = os.path.dirname(os.path.abspath(__file__))
with XDMFFile(comm, os.path.join(save_dir, "mesh_cones.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_tags, domain.geometry)
    xdmf.write_meshtags(facet_tags, domain.geometry)

print("Mesh sauvegardé : mesh_cones.xdmf")