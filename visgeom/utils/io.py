
""" IO handling of files. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import logging

import trimesh
import numpy as np
import nibabel as nib

def load_mesh(filename: str):
    """
    Load mesh data and convert to trimesh.
    Supported formats:
        - .ply
        - .stl
        - .gii
        - FreeSurfer file format

    :param filename: The mesh file to load.
    """
    logging.info("Loading file %s", filename)

    try:
        mesh = trimesh.load(filename, process=False)
    except ValueError:
        try:
            gii_mesh = nib.load(filename).agg_data()
            mesh = trimesh.Trimesh(gii_mesh[0], gii_mesh[1], process=False)
        except:
            geo = nib.freesurfer.io.read_geometry(filename)
            mesh = trimesh.Trimesh(geo[0], geo[1], process=False)

    return mesh

def load_vertex_values(filename: str):
    """ Load array data.
    Supported formats:
        - .npy
        - Something that nibabel can handle
        - FreeSurfer annot and morph files
    """
    try:
        data = np.load(filename)
    except ValueError:
        try:
            # data = nib.load(filename).agg_data()
            data = nib.load(filename).get_fdata().squeeze()
        except:
            data = nib.freesurfer.io.read_annot(filename)[0] if (
                "annot" in filename
            ) else nib.freesurfer.io.read_morph_data(filename)
    return data

def load_img3D(filename: str):
    """ Load 3D image from file with nibabel """
    return nib.load(filename)
