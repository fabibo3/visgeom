
""" Create a video with rotating meshes. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import sys
from argparse import ArgumentParser

import time
import trimesh
import numpy as np
import pyvista as pv
import nibabel.freesurfer.io as fsio
from tqdm import tqdm

title = "Cortical Thickness 20230513_123320s401a1004"
# title = "Pial"

fs_files = [
    "/mnt/ai-med-nas/Software/Freesurfer72/subjects/fsaverage/surf/lh.inflated",
    "/mnt/ai-med-nas/Software/Freesurfer72/subjects/fsaverage/surf/rh.inflated",
]

cthfile1 = "/mnt/bear/work/SUBJECTS_DIR/20230513_123320s401a1004/surf/lh.bi_thickness"
cthfile2 = "/mnt/bear/work/SUBJECTS_DIR/20230513_123320s401a1004/surf/rh.bi_thickness"

for fn in fs_files:
    v, f = fsio.read_geometry(fn)
    _ = trimesh.Trimesh(vertices=v, faces=f, process=False).export("/home/fabi/work/visgeom/inputs/" + fn.split('/')[-1] + ".ply")

file1 = "/home/fabi/work/visgeom/inputs/lh.inflated.ply"
file2 = "/home/fabi/work/visgeom/inputs/rh.inflated.ply"

def rot_mat(angle, axis):
    """ Get a rotation matrix from angle in rad """
    Rx = lambda x: np.array(
        [[1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]]
    )
    Ry = lambda x: np.array(
        [[np.cos(x), 0, np.sin(x)],
        [0, 1, 0],
        [-np.sin(x), 0, np.cos(x)]]
    )
    Rz = lambda x: np.array(
        [[np.cos(x), -np.sin(x), 0],
        [np.sin(x), np.cos(x), 0],
        [0, 0, 1]]
    )

    rot_mats = (Rx, Ry, Rz)

    return rot_mats[axis](angle)


def vis_mesh_rotating(
    files, cth_files, out_fn="video_out.mp4", rotation_axis=2
):
    """ Create a video with rotating meshes. """

    if out_fn != "video_out.mp4" and os.path.exists(out_fn):
        print("Path already exists!")
        sys.exit(0)

    # Custom theme
    pv.set_plot_theme('document')

    # Read input meshes
    cloud = pv.read(files[0])
    cloud.points[:, 0] -= 40 # For lh.inflated
    cloud['cth'] = fsio.read_morph_data(cth_files[0])
    for fn, cth_fn in zip(files[1:], cth_files[1:]):
        new_cloud = pv.read(fn)
        new_cloud.points[:, 0] += 40 # For rh.inflated
        new_cloud['cth'] = fsio.read_morph_data(cth_fn)
        cloud = cloud + new_cloud

    print(cloud)

    # Center w.r.t. rotation axis
    cloud.points[:, rotation_axis] -= cloud.points[:, rotation_axis].mean()

    # Seems to be cructial to add off_screen here although it's not in the
    # original example (https://github.com/pyvista/pyvista/issues/5130)
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(out_fn)

    plotter.add_mesh(
        cloud,
        smooth_shading=True,
        cmap='RdYlGn',
        scalars='cth',
        clim=[1, 4],
        scalar_bar_args={'title': 'cortical thickness (mm)'}
        # show_scalar_bar=False,
    )
    # plotter.add_mesh(cloud.outline_corners())
    # plotter.show_axes()

    # Rotate and write frames
    step_size = 2 # Degree
    rad_angle = 2 * np.pi * step_size / 360
    for i in tqdm(range(0, 360, step_size)):
        cloud.points = cloud.points @ rot_mat(rad_angle, rotation_axis).T
        plotter.write_frame()

    plotter.close()

    print("Stored video at ", out_fn)


if __name__ == '__main__':
    # conda env create -n visgeom_video python=3.9
    # pip install imageio[ffmpeg]==2.31.4
    # pip install pyvista
    # pip install trimesh nibabel tqdm
    print("Need environment 'visgeom_video'!")
    vis_mesh_rotating(
        [file1, file2],
        [cthfile1, cthfile2],
        out_fn="/home/fabi/work/visgeom/outputs/rotating_20230513_123320s401a1004.mp4"
    )
