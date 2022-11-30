""" Utility functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import numpy as np
from trimesh import Trimesh

def transform_mesh_affine(vertices: np.ndarray,
                          faces: np.ndarray,
                          transformation_matrix: np.ndarray):
    """ Transform vertices of shape (V, D) or (S, V, D) using a given
    transformation matrix such that v_new = (mat @ v.T).T. """

    ndims = vertices.shape[-1]
    if (tuple(transformation_matrix.shape) != (ndims + 1, ndims + 1)):
        raise ValueError("Wrong shape of transformation matrix.")

    coords = np.concatenate(
        (vertices.T, np.ones([1, vertices.shape[0]])), axis=0
    )

    # Transform
    new_coords = (transformation_matrix @ coords)

    # Adapt faces s.t. normal convention is still fulfilled
    if np.sum(np.sign(np.diag(transformation_matrix)) == -1) % 2 == 1:
        new_faces = np.flip(faces, axis=1)
    else: # No flip required
        new_faces = faces

    # Correct shape
    new_coords = new_coords.T[:,:-1]

    return new_coords, new_faces

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


def voxelize_mesh(vertices, faces, shape):
    """ Voxelize the mesh and return a segmentation map of 'shape' for each
    mesh class.

    :param vertices: The vertices of the mesh
    :param faces: Corresponding faces as indices to vertices
    :param shape: The shape the output image should have
    """
    assert len(shape) == 3, "Shape should be 3D"

    voxelized_mesh = np.zeros(shape, dtype=np.int32)
    pv = get_occupied_voxels(vertices, faces, shape)
    if pv is not None:
        # Occupied voxels belong to one class
        voxelized_mesh[pv[:, 0], pv[:, 1], pv[:, 2]] = 1
    else:
        # No mesh in the valid range predicted --> keep zeros
        pass

    return voxelized_mesh
