""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from typing import Union
from collections.abc import Sequence

import numpy as np
# import open3d as o3d # Leads to double logging, uncomment if needed
import pymeshlab as pyml
import nibabel as nib
import matplotlib.pyplot as plt
import trimesh
import torch
import matplotlib

from scipy.stats import zscore
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours

from utils.coordinate_transform import normalize_vertices_per_max_dim

def min_max_norm(volume: np.array):
    return (volume - volume.min()) / (volume.max() - volume.min())

def z_score_norm(volume: np.array):
    return zscore(volume, axis=None)

def find_label_to_img(base_dir: str, img_id: str, label_dir_id="label"):
    """
    Get the label file corresponding to an image file.
    Note: This is only needed in the case where the dataset is not represented
    by a data.dataset.

    :param str base_dir: The base directory containing the label directory.
    :param str img_id: The id of the image that is also cotained in the label
    file.
    :param str label_dir_id: The string that identifies the label directory.
    :return The label file name.
    """
    label_dir = None
    label_name = None
    for d in os.listdir(base_dir):
        d_full = os.path.join(base_dir, d)
        if (os.path.isdir(d_full) and (label_dir_id in d)):
            label_dir = d_full
            print(f"Found label directory '{label_dir}'.")
    if label_dir is None:
        print(f"No label directory found in {base_dir}, maybe adapt path"\
              " specification or search string.")
        return None
    # Label dir found
    for ln in os.listdir(label_dir):
        if img_id == ln.split('.')[0]:
            label_name = ln

    if label_name is None:
        print(f"No file with id '{img_id}' found in directory"\
              " '{label_dir}'.")
        return None
    return os.path.join(label_dir, label_name)


def show_pointcloud(filenames: Union[str, list], backend='open3d', opacity=1.0,
                    values=None, screenshot=None, clim=[0.0, 5.0]):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d or pyvista.

    :param str filenames: A list of files or a directory name.
    :param str backend: 'open3d' or 'pyvista' (default)
    """
    if values and backend == 'open3d':
        raise ValueError("Values not supported with open3d.")
    if isinstance(filenames, str):
        if os.path.isdir(filenames):
            path = filenames
            filenames = os.listdir(path)
            filenames.sort()
            filenames = [os.path.join(path, fn) for fn in filenames]
        else:
            filenames = [filenames]
    # filenames = sorted(filenames)

    if values:
        if isinstance(values, str):
            if os.path.isdir(values):
                path = values
                values = os.listdir(path)
                values.sort()
                values = [os.path.join(path, fn) for fn in values]
            else:
                values = [values]

        # values = sorted(values)

    for i, fn in enumerate(filenames):
        print(f"File: {fn}")
        if values:
            if len(values) != len(filenames):
                raise ValueError("Number of files should be equal to number"
                                 " of value files.")
            value = values[i]
            print(f"Value: {value}")
        else:
            value = None
        if backend == 'open3d':
            show_pointcloud_open3d(fn)
        elif backend == 'pyvista':
            show_pointcloud_pyvista(fn, opacity=opacity, valuefile=value,
                                    screenshot_fn=screenshot, clim=clim)
        else:
            raise ValueError("Unknown backend {}".format(backend))

def show_pointcloud_open3d(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d.
    An alternative is based on pyvista, see
    'show_pointcloud_pyvista'

    :param str filename: The file that should be visualized.
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    print(mesh)
    o3d.visualization.draw_geometries([mesh])

def get_color(filename: str):
    if "struc0" in filename:
        return 'mediumpurple'
    if "struc1" in filename:
        return "darkcyan"
    if "struc2" in filename:
        return "firebrick"
    if "struc3" in filename:
        return "darkgoldenrod"

    return "gray"

def store_with_color(t_mesh, values, cmap, path='/mnt/c/Users/Fabian/Desktop/c_mesh.ply'):
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

def show_pointcloud_pyvista(filename: str, opacity=1.0, valuefile=None,
                            screenshot_fn=None, clim=None):
    """
    Show a point cloud stored in a file (e.g. .ply) using pyvista.

    :param str filename: The file that should be visualized.
    """
    import pyvista as pv

    # Custom theme
    pv.set_plot_theme('document')

    try:
        mesh = trimesh.load(filename, process=False)
        # Reduce mesh
        # ms = pyml.MeshSet()
        # ms.add_mesh(pyml.Mesh(mesh.vertices, mesh.faces))
        # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=10000)
        # mesh = trimesh.Trimesh(ms.current_mesh().vertex_matrix(),
                               # ms.current_mesh().face_matrix())
    except ValueError:
        try:
            gii_mesh = nib.load(filename).agg_data()
            mesh = trimesh.Trimesh(gii_mesh[0], gii_mesh[1], process=False)
        except:
            geo = nib.freesurfer.io.read_geometry(filename)
            mesh = trimesh.Trimesh(geo[0], geo[1], process=False)
    # pyvista has different face format
    faces = np.hstack([
        np.ones([mesh.faces.shape[0], 1], dtype=int) * mesh.faces.shape[1],
        mesh.faces
    ])
    cloud = pv.PolyData(mesh.vertices, faces)
    print(cloud)

    color = get_color(filename)

    if ".ply" in filename:
        basename = os.path.basename(filename)
    elif ".stl" in filename:
        basename = os.path.basename(filename).replace("stl", "ply")
    else:
        basename = os.path.basename(filename) + ".ply"

    plotter = pv.Plotter()
    # if (
        # ("lh" in basename) or ("struc0" in basename) or ("struc2" in basename)
    # ):
        # # Side
        # # cpos = [(-418.25028611249786, 112.68252128002338, 127.91361821823739),
                 # # (-30.733123779296875, -17.3704833984375, 14.500244140625),
                 # # (0.3203420879962193, 0.1371195047899042, 0.9373255507369859)]
        # # Frontal
        # cpos = [(-418.25028611249786, 112.68252128002338, 127.91361821823739),
                # (-30.733123779296875, -17.3704833984375, 14.500244140625),
                # (0.3203420879962193, 0.1371195047899042, 0.9373255507369859)]
        # plotter.camera_position = cpos
    # if (
        # ("rh" in basename) or ("struc2" in basename) or ("struc3" in basename)
    # ):
        # # Frontal
        # cpos = [(508.2274696328591, 29.2196305896372, 139.13991053647004),
                # (0.0, 0.0, 0.0),
                # (-0.26927015852997643, 0.1670220907108095, 0.9484709816013517)]
        # plotter.camera_position = cpos

    print("Camera:")
    print(plotter.camera_position)
    plotter.set_background(color='white')
    if valuefile is None:
        plotter.add_mesh(
            cloud,
            opacity=opacity,
            smooth_shading=True,
            specular=0.5,
            # show_edges=True,
        )
        plotter.add_title(f"{filename}", font_size=6)
    else:
        try:
            value = np.load(valuefile)
        except:
            try:
                value = nib.load(valuefile).agg_data()
            except:
                value = nib.freesurfer.io.read_annot(valuefile)[0] if (
                    valuefile.split(".")[-1] == "annot"
                ) else nib.freesurfer.io.read_morph_data(valuefile)

        if clim is None:
            clim = [np.min(value), np.max(value)]

        # Avoid negative colors
        # value = value + 1
        # Discrete colormap for parcellation
        # cmap = plt.get_cmap('tab20b', np.max(value) - np.min(value))
        cmap = plt.get_cmap("jet")
        # cmap = plt.get_cmap("Wistia")
        # norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
        # cmap.set_bad('white',1.)
        plotter.add_title(f"{filename}\nColor: {valuefile}", font_size=6)
        plotter.add_mesh(
            cloud,
            opacity=opacity,
            smooth_shading=True,
            specular=0.5,
            cmap=cmap,
            scalars=value,
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
        # Store mesh with color s.t. it can be opened in MeshLab
        # store_with_color(
            # mesh,
            # # Normalize values to the same range as the shown plot
            # # norm(value),
            # value,
            # cmap,
            # path=os.path.join(
                # "/mnt/c/Users/Fabian/Desktop/", basename
            # )
        # )

    points = cloud.points[:10]
    labels = [str(i) for i in range(10)]
    # plotter.add_point_labels(points, labels)
    screenshot_fn = "/mnt/c/Users/Fabian/Desktop/" + basename.replace(".ply", ".png")
    if screenshot_fn:
        plotter.show(screenshot=screenshot_fn)
        print("Stored a screenshot at ", screenshot_fn)
    else:
        cpos = plotter.show(interactive=True, auto_close=True, return_cpos=True)
        print(cpos)
        plotter.close()

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

