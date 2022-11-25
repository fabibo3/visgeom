#!/usr/bin/env python3

""" Visualization of 3D data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

from argparse import ArgumentParser
from utils.visualization import show_pointcloud

def vis_mesh():
    parser = ArgumentParser(description="Visualize 3D meshes.")
    parser.add_argument('filenames',
                        nargs='+',
                        type=str,
                        help="The filenames or the name of one folder to visualize.")
    parser.add_argument('--values',
                        nargs='*',
                        type=str,
                        help="Values to map on vertices.")
    parser.add_argument('--all_values',
                        default=None,
                        type=str,
                        help="Values to map on vertices of each mesh.")
    parser.add_argument('--clone',
                        action="store_true",
                        help="Meshes to map all values onto.")
    parser.add_argument('--opacity',
                        type=float,
                        default=1.0,
                        help="Opacity used for rendering.")
    parser.add_argument('--clim',
                        type=float,
                        default=None,
                        nargs=2,
                        help="Color limits for value visualization.")
    parser.add_argument('--screenshot',
                        type=str,
                        default=None,
                        help="Optionally specify a path where a screenshot is"
                        " stored.")
    parser.add_argument('--backend',
                        metavar='LIB',
                        type=str,
                        default='pyvista',
                        help="The library used for visualization, 'open3d' or 'pyvista' (default).")

    args = parser.parse_args()

    if os.path.isdir(args.filenames[0]):
        filenames = args.filenames[0]
    else:
        filenames = args.filenames

    # Use the same value file for all meshes
    if args.all_values is not None:
        args.values = len(filenames) * [args.all_values]

    if args.clone:
        filenames = filenames * len(args.values)


    show_pointcloud(filenames, backend=args.backend, opacity=args.opacity,
                    screenshot=args.screenshot,
                    values=args.values, clim=args.clim)


if __name__ == "__main__":
    vis_mesh()
