""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import logging
from collections.abc import Sequence
from typing import Union, List

import numpy as np
# import open3d as o3d # Leads to double logging, uncomment if needed
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import trimesh
import pyvista as pv
from scipy.stats import zscore
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours

from utils.utils import voxelize_mesh, transform_mesh_affine

module_dir = os.path.dirname(os.path.abspath(__file__))

# Define the image slices to show, e.g. 0.5 means that the slice in the
# middle of the image is shown
slice_x = 0.25
slice_y = 0.5
slice_z = 0.5


def min_max_norm(volume: np.array):
    return (volume - volume.min()) / (volume.max() - volume.min())

def z_score_norm(volume: np.array):
    return zscore(volume, axis=None)

def store_with_color(t_mesh, values, cmap, path):
    """ Store a mesh with colors per vertex. Note that values should be in the
    range [0,1], e.g., normalized in advance with
    matplotlib.colors.Normalize!!!
    """
    colors = cmap(values)
    t_mesh.visual.vertex_colors = colors
    # t_mesh.visual.vertex_colors[larger_vmax] = [100, 100, 100, 250]
    t_mesh.export(path, file_type='ply')

    # cmap = cm.get_cmap(cmap_str, 2)    # PiYG

    # for i in range(cmap.N):
        # rgba = cmap(i)
        # # rgb2hex accepts rgb or rgba
        # print(matplotlib.colors.rgb2hex(rgba))

def vis_mesh(mesh: trimesh.Trimesh,
             vertex_values=None,
             screenshot=None,
             clim=None,
             title="Pyvista plot",
             cpos=None,
             point_labels=False,
             # cmap_name="autumn", # For group study
             # cmap_name="RdYlGn_r", # For curvature
             cmap_name="jet",
             # cmap_name='tab20b', # Parcellation
             interactive_cpos=True):
    """
    Show trimesh meshes with pyvista and optionally map values onto the
    vertices using a matplotlib colormap.
    """

    # Custom theme
    pv.set_plot_theme('document')

    if clim is None:
        clim = [np.min(vertex_values), np.max(vertex_values)]

    # pyvista has different face format
    faces = np.hstack([
        np.ones([mesh.faces.shape[0], 1], dtype=int) * mesh.faces.shape[1],
        mesh.faces
    ])
    cloud = pv.PolyData(mesh.vertices, faces)
    print(cloud)
    plotter = pv.Plotter()
    plotter.add_title(title, font_size=6)

    if cpos is not None:
        plotter.camera_position = cpos
    logging.debug("Camera: ")
    logging.debug(plotter.camera_position)
    plotter.set_background(color='white')
    if vertex_values is None:
        plotter.add_mesh(
            cloud,
            smooth_shading=True,
            specular=0.5,
            # show_edges=True,
        )
    else:
        logging.info("Unique vertex values:")
        logging.info(np.unique(vertex_values))
        cmap = plt.get_cmap(cmap_name)
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])

        plotter.add_mesh(
            cloud,
            smooth_shading=True,
            specular=0.5,
            cmap=cmap,
            scalars=vertex_values.copy(), # Plotter seems to change values sometimes
            clim=clim,
            # below_color='gray',
            # rgb=True,
            # clim=(np.nanmin(value), np.nanmax(value)),
            # scalars=1-value,
        )
        # value[value < 0.01] = 0.01
        # value[np.logical_and(value < 0.05, value > 0.01)] = 0.05
        # value[~np.isin(value, (0.01, 0.05))] = 0
        # plotter.add_mesh(
            # cloud,
            # opacity=opacity,
            # smooth_shading=True,
            # cmap=['blue', 'red', 'yellow'],
            # scalar_bar_args={"n_labels": 3},
            # scalars=value,
            # # clim=[0, 2],
        # )

    if point_labels:
        points = cloud.points[:10]
        labels = [str(i) for i in range(10)]
        plotter.add_point_labels(points, labels)
    if screenshot:
        # Store mesh with color s.t. it can be opened in MeshLab
        if vertex_values is not None:
            store_with_color(
                mesh,
                norm(vertex_values), # Normalize values to the same range as the shown plot
                cmap,
                path=screenshot.replace("png", "ply")
            )
        else:
            mesh.export(screenshot.replace("png", "ply"), file_type='ply')
        # Iterate through all meshes without waiting for user
        # plotter.show(screenshot=screenshot, interactive=False, auto_close=True)
        # Wait for user to close window
        plotter.show(screenshot=screenshot)
        logging.info("Stored a screenshot at %s", screenshot)
    elif interactive_cpos:
        cpos = plotter.show(interactive=True, auto_close=True, return_cpos=True)
        np.save(os.path.join(module_dir, "../cposes/last_cpos.npy"), cpos)
        plotter.close()
    else:
        plotter.show()

def _extract_slices(img3D):
    img1 = img3D
    img1 = img1[int(img3D.shape[0] * slice_x), :, :]
    img1 = np.flip(np.rot90(img1), axis=1)
    img2 = img3D
    img2 = img2[:, int(img3D.shape[1] * slice_y), :]
    img2 = np.rot90(img2)
    img3 = img3D
    img3 = img3[:, :, int(img3D.shape[2] * slice_z)]
    img3 = np.rot90(img3)

    return [img1, img2, img3]

def vis_img_slices(img: nib.Nifti1Image,
                   label: Union[nib.Nifti1Image, trimesh.Trimesh],
                   label_mode='fill',
                   output_file=None,
                   norm=None,
                   title="Slices"):
    """
    Show three centered slices of a 3D image and a histogram below with
    potential labels from mesh or voxel segmentation.

    :param img: The image to show
    :param label: A sequence of image or mesh labels
    :param label_mode: Either 'contour' or 'fill'
    :param output_dir: Optionally specify an output file.
    :param norm: Optionally normalize the image with one of the norms supported
    by 'supported_img_norms'
    """

    img3D = img.get_fdata()
    if norm is not None:
        img3D = supported_img_norms[norm](img3D)

    img1, img2, img3 = _extract_slices(img3D)

    if label is not None:
        try:
            # Mesh labels
            label.vertices, label.faces = transform_mesh_affine(
                label.vertices, label.faces, np.linalg.inv(img.affine)
            )
            label = _extract_slices(
                voxelize_mesh(
                    label.vertices, label.faces, img3D.shape
                )
            )
        except AttributeError as e:
            logging.debug(e)
            # Voxel label
            label = _extract_slices(label.get_fdata())

    _show_slices(
        [img1, img2, img3],
        labels=label,
        save_path=output_file,
        label_mode=label_mode,
        whole_volume=img3D,
        name=title
    )

def _show_slices(slices, labels, save_path, label_mode, whole_volume, name):
    """
    Visualize image slices in a row. If whole_volume is given, a histogram is
    also computed from it.

    """

    n_rows = 1 if whole_volume is None else 2
    fig, axs = plt.subplots(n_rows, len(slices))
    if len(slices) == 1:
        axs = [axs]

    for i, s in enumerate(slices):
        axs[0, i].imshow(s, cmap="gray")

    if labels is not None:
        for i, l in enumerate(labels):
            l = l.astype(np.uint8)
            if len(np.unique(l)) > 2:
                print(f"Available labels: {np.unique(l)}, please choose one:")
                x = input()
                l[l != int(x)] = 0
            l[l != 0] = 1

            if label_mode == 'fill':
                axs[0, i].imshow(l, cmap="Reds", alpha=0.3)

            elif label_mode == 'contour':
                contours = find_contours(l, np.max(l)/2)
                for c in contours:
                    axs[0, i].plot(c[:, 1], c[:, 0], linewidth=0.5,
                                   color='red')
            else:
                raise ValueError(f"Unknown label mode '{label_mode}'")

    # Histogram
    if whole_volume is not None:
        gs = axs[1, 0].get_gridspec()
        for ax in axs[1, :]:
            ax.remove()
        axbig = fig.add_subplot(gs[1, :])
        axbig.set_title("Histogram of volume")
        axbig.hist(whole_volume.flatten())

    fig.suptitle(name)
    fig.tight_layout()

    # save_path = "/mnt/c/Users/Fabian/Desktop/" + name.replace(".nii.gz", ".png")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def show_img_with_contour(img, vertices, edges, save_path=None):
    if vertices.ndim != 2 or edges.ndim != 2:
        raise ValueError("Vertices and edges should be in packed"
                         " representation.")
    plt.imshow(img, cmap="gray")
    vertices_edges = vertices[edges]

    plt.plot(vertices_edges[:,0,1], vertices_edges[:,0,0], color="red",
             marker='x', markeredgecolor="gray", markersize=1, linewidth=1)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

supported_img_norms = {
    'min_max': min_max_norm,
    'z_score': z_score_norm,
}

