""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import logging
from collections.abc import Sequence

import numpy as np
# import open3d as o3d # Leads to double logging, uncomment if needed
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import trimesh
import torch
import pyvista as pv

from scipy.stats import zscore
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours

from utils.coordinate_transform import normalize_vertices_per_max_dim

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
             cmap_name="jet", #tab20, Wistia
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
    plotter = pv.Plotter()
    plotter.add_title(title, font_size=6)

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
        cmap = plt.get_cmap(cmap_name)
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
        plotter.add_mesh(
            cloud,
            smooth_shading=True,
            specular=0.5,
            cmap=cmap,
            scalars=vertex_values,
            clim=clim,
            # above_color='red',
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
        store_with_color(
            mesh,
            norm(vertex_values), # Normalize values to the same range as the shown plot
            cmap,
            path=screenshot.replace("png", "ply")
        )
        plotter.show(screenshot=screenshot)
        logging.info("Stored a screenshot at %s", screenshot)
    elif interactive_cpos:
        cpos = plotter.show(interactive=True, auto_close=True, return_cpos=True)
        logging.info(cpos)
        plotter.close()
    else:
        plotter.show()

def show_img_slices_3D(filenames: str, show_label=True, dataset="Cortex",
                       label_mode='contour', labels_from_mesh: str=None,
                       output_file=None, voxel_label=None, norm=None):
    """
    Show three centered slices of a 3D image and a histogram below.

    :param str filenames: A list of files or a directory name.
    :param bool show_label: Try to find label corresponding to image and show
    image and label together if possible.
    :param dataset: Either 'Hippocampus' or 'Cortex'
    :param label_mode: Either 'contour' or 'fill'
    :param labels_from_mesh: Path to a mesh that is used as mesh label then.
    :param output_dir: Optionally specify an output file.
    :param voxel_label: Optionally provide a voxel label file.
    """
    # Define the image slices to show, e.g. 0.5 means that the slice in the
    # middle of the image is shown
    slice_x = 0.25
    slice_y = 0.5
    slice_z = 0.5

    if isinstance(filenames, str):
        if os.path.isdir(filenames):
            path = filenames
            filenames = os.listdir(path)
            filenames.sort()
            filenames = [os.path.join(path, fn) for fn in filenames]
        else:
            filenames = [filenames]

    for fn in filenames:
        img3D = nib.load(fn)
        print(f"Loading image {fn}...")
        assert img3D.ndim == 3, "Image dimension not equal to 3."

        img3D = img3D.get_fdata()
        if norm is not None:
            img3D = supported_img_norms[norm](img3D)

        img1 = img3D
        img1 = img1[int(img3D.shape[0] * slice_x), :, :]
        img1 = np.flip(np.rot90(img1), axis=1)
        img2 = img3D
        img2 = img2[:, int(img3D.shape[1] * slice_y), :]
        img2 = np.rot90(img2)
        img3 = img3D
        img3 = img3[:, :, int(img3D.shape[2] * slice_z)]
        img3 = np.rot90(img3)

        try:
            labels = _get_labels_from_mesh(
                labels_from_mesh,
                patch_size=img3D.shape,
                slices=[slice_x, slice_y, slice_z]
            )
        except ValueError:
            labels = None

        if labels is not None and show_label:
            # Read and show ground truth
            show_slices([img1, img2, img3], labels=labels,
                        label_mode=label_mode, save_path=output_file,
                        whole_volume=img3D, name=fn.split("/")[-1])

        else:
            show_slices([img1, img2, img3], save_path=output_file,
                        whole_volume=img3D, name=fn.split("/")[-1])

def _get_labels_from_mesh(mesh_labels, patch_size, slices):
    """ Generate voxel labels from mesh prediction(s)."""

    # Mesh processing requires pytorch3d
    from utils.utils import voxelize_mesh

    if not isinstance(mesh_labels, Sequence):
        mesh_labels = [mesh_labels]

    label1, label2, label3 = [], [], []
    for ml in mesh_labels:
        # trimesh.load does not distinguish structures in mesh
        mesh = trimesh.load(ml)
        vertices = torch.from_numpy(mesh.vertices) # Vx3
        faces = torch.from_numpy(mesh.faces) # Fx3
        # Potentially normalize: if the mean of all vertex coordinates is > 2,
        # it is assumed that the coordinates are not normalized
        if vertices.mean() > 2:
            vertices = normalize_vertices_per_max_dim(vertices, patch_size)

        voxelized = voxelize_mesh(vertices, faces, patch_size, 1).squeeze().cpu().numpy()
        label1.append(voxelized[int(patch_size[0] * slices[0]), :, :])
        label2.append(voxelized[:, int(patch_size[1] * slices[1]), :])
        label3.append(voxelized[:, :, int(patch_size[2] * slices[2])])

    return [label1, label2, label3]


def show_slices(slices, labels=None, save_path=None, label_mode='contour',
                whole_volume=None, name=None):
    """
    Visualize image slices in a row. If whole_volume is given, a histogram is
    also compute from it.

    :param array-like slices: The image slices to visualize.
    :param array-like labels (optional): The image segmentation label slices.
    """

    assert label_mode in ('contour', 'fill')
    colors = ('blue', 'green', 'cyan', 'yellow')

    # TMP
    slices = [slices[0]]
    whole_volume = None

    n_rows = 1 if whole_volume is None else 2
    fig, axs = plt.subplots(n_rows, len(slices))
    if len(slices) == 1:
        axs = [axs]

    for i, s in enumerate(slices):
        axs[0].imshow(s, cmap="gray")
        # axs[0, i].imshow(s, cmap="gray")

    if labels is not None:
        for i, l in enumerate(labels):
            if not isinstance(l, Sequence):
                l_ = [l]
            else:
                l_ = l

            for ll, col in zip(l_, colors):
                if label_mode == 'fill':
                    axs[0, i].imshow(ll, cmap="OrRd", alpha=0.3)
                else:
                    contours = find_contours(ll, np.max(ll)/2)
                    for c in contours:
                        axs[0, i].plot(c[:, 1], c[:, 0], linewidth=0.5,
                                    color=col)

    # Histogram
    if whole_volume is not None:
        gs = axs[1, 0].get_gridspec()
        for ax in axs[1, :]:
            ax.remove()
        axbig = fig.add_subplot(gs[1, :])
        axbig.set_title("Histogram of volume")
        axbig.hist(whole_volume.flatten())

    if name is not None:
        fig.suptitle(name)
    else:
        plt.suptitle("Image Slices")

    fig.tight_layout()

    # save_path = "/mnt/c/Users/Fabian/Desktop/" + name.replace(".nii.gz", ".png")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

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

def show_difference(img_1, img_2, save_path=None):
    """
    Visualize the difference of two 3D images in the center axes.

    :param array-like img_1: The first image
    :param array-like img_2: The image that should be compared to the first one
    :param save_path: Where the image is exported to
    """
    shape_1 = img_1.shape
    img_1_slices = [img_1[shape_1[0]//2, :, :],
                    img_1[:, shape_1[1]//2, :],
                    img_1[:, :, shape_1[2]//2]]
    shape_2 = img_2.shape
    assert shape_1 == shape_2, "Compared images should be of same shape."
    img_2_slices = [img_2[shape_2[0]//2, :, :],
                    img_2[:, shape_2[1]//2, :],
                    img_2[:, :, shape_2[2]//2]]
    diff = [(i1 != i2).long() for i1, i2 in zip(img_1_slices, img_2_slices)]

    fig, axs = plt.subplots(1, len(img_1_slices))
    if len(img_1_slices) == 1:
        axs = [axs]

    for i, s in enumerate(img_1_slices):
        axs[i].imshow(s, cmap="gray")

    for i, (l, ax) in enumerate(zip(diff, axs)):
        im = ax.imshow(l, cmap="OrRd", alpha=0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    fig.tight_layout()

    plt.suptitle("Difference")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


supported_img_norms = {
    'min_max': min_max_norm,
    'z_score': z_score_norm,
}

