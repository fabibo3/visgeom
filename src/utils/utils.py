""" Utility functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import numpy as np
import torch
import torch.nn.functional as F
from trimesh import Trimesh

from utils.coordinate_transform import (
    unnormalize_vertices_per_max_dim
)


def sample_inner_volume_in_voxel(volume):
    """ Samples an inner volume in 3D given a volume representation of the
    objects. This can be seen as 'stripping off' one layer of pixels.

    Attention: 'sample_inner_volume_in_voxel' and
    'sample_outer_surface_in_voxel' are not inverse to each other since
    several volumes can lead to the same inner volume.
    """
    neg_volume = -1 * volume # max --> min
    neg_volume_a = F.pad(neg_volume, (0, 0, 0, 0, 1, 1)) # Zero-pad
    a = F.max_pool3d(neg_volume_a[None, None].float(), kernel_size=(3, 1, 1), stride=1)[0]
    neg_volume_b = F.pad(neg_volume, (0, 0, 1, 1, 0, 0)) # Zero-pad
    b = F.max_pool3d(neg_volume_b[None, None].float(), kernel_size=(1, 3, 1), stride=1)[0]
    neg_volume_c = F.pad(neg_volume, (1, 1, 0, 0, 0, 0)) # Zero-pad
    c = F.max_pool3d(neg_volume_c[None, None].float(), kernel_size=(1, 1, 3), stride=1)[0]
    border, _ = torch.max(torch.cat([a, b, c], dim=0), dim=0)
    border = -1 * border
    inner_volume = torch.logical_and(volume, border)
    # Seems to lead to problems if volume.dtype == torch.uint8
    return inner_volume.type(volume.dtype)


def get_occupied_voxels(vertices, faces, shape):
    """Get the occupied voxels of the mesh lying within 'shape'.

    Attention: 'shape' should be defined in the same coordinte system as
    the mesh.
    """
    assert len(shape) == 3, "Shape should represent 3 dimensions."

    voxelized = Trimesh(vertices, faces, process=False).voxelized(1.0).fill()
    # Coords = trimesh coords + translation
    vox_occupied = np.around(voxelized.sparse_indices +\
        voxelized.translation).astype(int)

    # 0 <= coords < shape
    vox_occupied = np.asarray(vox_occupied)
    mask = np.ones((vox_occupied.shape[0]), dtype=bool)
    for i, s in enumerate(shape):
        in_box = np.logical_and(vox_occupied[:, i] >= 0,
                                vox_occupied[:, i] < s)
        mask = np.logical_and(mask, in_box)
    vox_occupied = vox_occupied[mask]

    if vox_occupied.size < 1:
        # No occupied voxels in the given shape
        vox_occupied = None

    return vox_occupied


def voxelize_mesh(vertices, faces, shape, n_m_classes, strip=True):
    """ Voxelize the mesh and return a segmentation map of 'shape' for each
    mesh class.

    :param vertices: The vertices of the mesh
    :param faces: Corresponding faces as indices to vertices
    :param shape: The shape the output image should have
    :param n_m_classes: The number of mesh classes, i.e., the number of
    different structures in the mesh. This is currently ignored but should be
    implemented at some time.
    :param strip: Whether to strip the outer layer of the voxelized mesh. This
    is often a more accurate representation of the discrete volume occupied by
    the mesh.
    """
    assert len(shape) == 3, "Shape should be 3D"
    assert (n_m_classes == vertices.shape[0] == faces.shape[0]
            or (n_m_classes == 1 and vertices.ndim == faces.ndim == 2)),\
            "Wrong shape of vertices and/or faces."

    voxelized_mesh = torch.zeros(shape, dtype=torch.long)
    vertices = vertices.view(n_m_classes, -1, 3)
    faces = faces.view(n_m_classes, -1, 3)
    unnorm_verts = unnormalize_vertices_per_max_dim(
        vertices.view(-1, 3), shape
    ).view(n_m_classes, -1, 3)
    voxelized_all = []
    for v, f in zip(unnorm_verts, faces):
        vm = voxelized_mesh.clone()
        pv = get_occupied_voxels(v, f, shape)
        if pv is not None:
            # Occupied voxels belong to one class
            vm[pv[:, 0], pv[:, 1], pv[:, 2]] = 1
        else:
            # No mesh in the valid range predicted --> keep zeros
            pass

        # Strip outer layer of voxelized mesh
        if strip:
            vm = sample_inner_volume_in_voxel(vm)

        voxelized_all.append(vm)

    return torch.stack(voxelized_all)
