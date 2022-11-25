
""" Visualize cortical thickness. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import trimesh
import torch
import numpy as np
import pyvista as pv
from pytorch3d.ops import laplacian

from utils.mesh import curv_from_cotcurv_laplacian

# Load meshes
# mesh_1 = pv.read("/home/fabi/remote/experiments/lrz/exp_2/test_template_168058_OASIS/meshes/OAS1_0006_MR1_epoch95_struc1_meshpred.ply")
# mesh_2 = pv.read("/home/fabi/remote/experiments/lrz/exp_2/test_template_168058_OASIS/meshes/OAS1_0006_MR1_epoch95_struc3_meshpred.ply")
mesh_1 = pv.read("/home/fabi/work/tmp/rh_white.ply")
mesh_2 = pv.read("/home/fabi/work/tmp/rh_pial.ply")

# Mesh
verts_1 = mesh_1.points
# Faces in pyvista are stored in the format
# [num. vertices in f1, v1, v2, ..., num. vertices in f2, ...]
faces_mask_1 = np.ones_like(mesh_1.faces, dtype=bool)
faces_mask_1[::4] = False
faces_1 = mesh_1.faces[faces_mask_1].reshape((-1, 3))
tri_mesh_1 = trimesh.Trimesh(verts_1, faces_1, process=False)
#
verts_2 = mesh_2.points
# Faces in pyvista are stored in the format
# [num. vertices in f2, v2, v2, ..., num. vertices in f2, ...]
faces_mask_2 = np.ones_like(mesh_2.faces, dtype=bool)
faces_mask_2[::4] = False
faces_2 = mesh_2.faces[faces_mask_2].reshape((-1, 3))
tri_mesh_2 = trimesh.Trimesh(verts_2, faces_2, process=False)
#
pcl_1 = verts_1
pcl_2 = verts_2

# Compute point to mesh distances and metrics
_, thickness_1, _ = trimesh.proximity.closest_point(tri_mesh_2, pcl_1)
_, thickness_2, _ = trimesh.proximity.closest_point(tri_mesh_1, pcl_2)

np.save("../misc/thickness_1.npy", thickness_1)
np.save("../misc/thickness_2.npy", thickness_2)

print(f"Average thickness 1: {np.mean(thickness_1)}")
print(f"Average thickness 2: {np.mean(thickness_2)}")

# Display
pv.set_plot_theme('doc')
plotter_1 = pv.Plotter()
plotter_1.add_mesh(
    mesh_1,
    scalars=thickness_1,
    cmap='plasma',
    clim=[0, 5],
    smooth_shading=True
)

fn = "/mnt/c/Users/Fabian/Desktop/experiments/thickness_demo_OASIS_0006/thickness_3.png"
plotter_1.show(screenshot=fn)
print("Stored screenshot at ", fn)
#
plotter_2 = pv.Plotter()
plotter_2.add_mesh(
    mesh_2,
    scalars=thickness_2,
    cmap='plasma',
    clim=[0, 5],
    smooth_shading=True
)

fn = "/mnt/c/Users/Fabian/Desktop/experiments/thickness_demo_OASIS_0006/thickness_4.png"
plotter_2.show(screenshot=fn)
print("Stored screenshot at ", fn)

