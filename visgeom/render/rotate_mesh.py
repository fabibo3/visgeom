
""" Create a video with rotating meshes. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import torch
import numpy as np
import pyvista as pv
# from pytorch3d.ops import knn_points

title = "White Matter"
# title = "Pial"

# file1 = "/home/fabi/remote/experiments/lrz/exp_2/test_template_168058_OASIS/meshes/OAS1_0044_MR1_epoch95_struc0_meshpred.ply"
# file2 = "/home/fabi/remote/experiments/lrz/exp_2/test_template_168058_OASIS/meshes/OAS1_0044_MR1_epoch95_struc1_meshpred.ply"
file1 = "/mnt/ai-med-nas/Projects/Vox2Cortex/V2C-Flow/v2c-flow_experiments/lrz_exp_211_extended_2/test_template_fsaverage-smooth-no-parc_ADNI_CSR_large_n_5/meshes/32409_epoch102_struc0_meshpred.ply"
file2 = "/mnt/ai-med-nas/Projects/Vox2Cortex/V2C-Flow/v2c-flow_experiments/lrz_exp_211_extended_2/test_template_fsaverage-smooth-no-parc_ADNI_CSR_large_n_5/meshes/32409_epoch102_struc1_meshpred.ply"

# gtfile1 = "/home/fabi/remote_data/OASIS/CSR_data/OAS1_0044_MR1/lh_white.ply"
# gtfile2 = "/home/fabi/remote_data/OASIS/CSR_data/OAS1_0044_MR1/rh_white.ply"

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

def compare_meshes(pntcld1, pntcld2):
    """ Compare each point of pointclound 1 to its closest neighbor in
    pointcloud 2 and return the distances. """
    pntcld1 = torch.tensor(pntcld1)
    pntcld2 = torch.tensor(pntcld2)
    _, _, nn = knn_points(pntcld1[None], pntcld2[None], K=1, return_nn=True)
    error = np.linalg.norm(pntcld1 - nn.squeeze().cpu().numpy(), axis=1)

    return error

def vis_mesh_rotating(
    files, gtfiles=None, out_fn="../misc/video.mp4", rotation_axis=2
):
    """ Create a video with rotating meshes. """

    # Custom theme
    pv.set_plot_theme('document')

    # Read input meshes
    cloud = pv.read(files[0])
    for fn in files[1:]:
        cloud = cloud + pv.read(fn)

    print(cloud)

    # Compare to ground truth
    if gtfiles is not None:
        gtcloud = pv.read(gtfiles[0])
        for fn in gtfiles[1:]:
            gtcloud = gtcloud + pv.read(fn)
        error = compare_meshes(cloud.points, gtcloud.points)

    # Center w.r.t. rotation axis
    cloud.points[:, rotation_axis] -= cloud.points[:, rotation_axis].mean()

    plotter = pv.Plotter()
    plotter.open_movie(out_fn)
    plotter.add_title(title)

    if gtfiles is None:
        plotter.add_mesh(cloud, smooth_shading=True, color='paleturquoise')
    else:
        plotter.add_mesh(
            cloud,
            smooth_shading=True,
            scalars=error,
            cmap='YlOrRd',
            clim=[0, 5],
            scalar_bar_args={'title': 'Error w.r.t. FreeSurfer'}
        )

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
        # [gtfile1, gtfile2],
        None,
        "/mnt/c/Users/Fabian/Desktop/video.mp4"
    )
