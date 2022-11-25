
""" Convenience script to visualize freesurfer thickness """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import numpy as np
import trimesh
from trimesh.base import Trimesh
from nibabel.freesurfer.io import read_morph_data, read_geometry

from utils.mesh import Mesh

argparser = ArgumentParser(description="Visualize a mesh together with"
                           " per-vertex morphology, e.g., thickness.")
argparser.add_argument("MESH_NAME",
                       type=str,
                       help="The input mesh file name.")
argparser.add_argument("MORPH_NAME",
                       type=str,
                       help="The morphology file name.")
argparser.add_argument("OUT_FILE",
                       type=str,
                       help="The output file name.")
args = argparser.parse_args()
mesh_fn= args.MESH_NAME
morph_fn= args.MORPH_NAME
out_fn = args.OUT_FILE

# Use nibabel.freesurfer.io.read_geometry to load mesh since lh_pial.stl etc.
# contain duplicate vertices
try:
    vertices, faces = read_geometry(mesh_fn)
    mesh = Trimesh(vertices, faces, process=False)
except ValueError: # Maybe it's a mesh file? --> try anyways
    mesh = trimesh.load(mesh_fn)

morphology = read_morph_data(morph_fn)

# Store
Mesh(
    mesh.vertices, mesh.faces, features=morphology
).store_with_features(out_fn)

print("Average morphology falue: ", str(np.mean(morphology)))
