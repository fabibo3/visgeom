
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import matplotlib.pyplot as plt
import trimesh
import matplotlib
import numpy as np

from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

cmap_str = 'autumn_r'
vmin = 1e-7
vmax = 0.05

mesh_fn = "/mnt/c/Users/Fabian/Desktop/c_mesh_ours_smoothed.ply"
values_fn = "/home/fabi/remote/experiments/exp_576_v2/test_template_168058_ADNI_CSR_large/group_analysis/p_thickness_rh_pial_ours.npy"
out_mesh_fn = "/mnt/c/Users/Fabian/Desktop/c_mesh_ours_smoothed_background.ply"

t_mesh = trimesh.load(mesh_fn, process=False)

values = np.load(values_fn)
larger_vmax = values > vmax

autumn = cm.get_cmap(cmap_str)
color_norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
colors = autumn(color_norm(values))
t_mesh.visual.vertex_colors[larger_vmax] = [100, 100, 100, 250]
t_mesh.export(out_mesh_fn)
