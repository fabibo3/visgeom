
""" Create a video with rotating meshes. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import time
import trimesh
import numpy as np
import pyvista as pv
import nibabel.freesurfer.io as fsio

title = "Cortical Thickness"
# title = "Pial"

fs_files = [
    "/mnt/ai-med-nas/Data_Neuro/ADNI_SEG/no_reg/FS72/223011/surf/lh.inflated",
    "/mnt/ai-med-nas/Data_Neuro/ADNI_SEG/no_reg/FS72/223011/surf/rh.inflated"
]

cthfile1 = "/mnt/ai-med-nas/Data_Neuro/ADNI_SEG/no_reg/FS72/223011/surf/lh.thickness"
cthfile2 = "/mnt/ai-med-nas/Data_Neuro/ADNI_SEG/no_reg/FS72/223011/surf/rh.thickness"

for fn in fs_files:
    v, f = fsio.read_geometry(fn)
    _ = trimesh.Trimesh(vertices=v, faces=f, process=False).export("/home/fabi/work/med-vis/inputs/" + fn.split('/')[-1] + ".ply")

file1 = "/home/fabi/work/med-vis/inputs/lh.inflated.ply"
file2 = "/home/fabi/work/med-vis/inputs/rh.inflated.ply"

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

    plotter = pv.Plotter()
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
    for i in range(0, 360, step_size):
        rad_angle = 2 * np.pi * step_size / 360
        cloud.points = cloud.points @ rot_mat(rad_angle, rotation_axis).T
        plotter.write_frame()

    plotter.close()

    print("Stored video at ", out_fn)


if __name__ == '__main__':
    vis_mesh_rotating(
        [file1, file2],
        [cthfile1, cthfile2],
        out_fn="/home/fabi/work/med-vis/outputs/rotating_cth.mp4"
    )
