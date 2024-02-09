#!/usr/bin/env python3

""" Visualization of 3D medical data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from argparse import ArgumentParser

import numpy as np

from utils.visualization import vis_mesh, vis_img_slices
from utils.io import (
    load_mesh,
    load_vertex_values,
    load_img3D,
)

import logging
logging.basicConfig(level=logging.DEBUG)

def main():
    parser = ArgumentParser(description="Visualize 3D meshes.")
    parser.add_argument('-m', '--meshes',
                        nargs='+',
                        type=str,
                        default=None,
                        help="The meshes to visualize.")
    parser.add_argument('-i', '--images',
                        nargs='+',
                        type=str,
                        default=None,
                        help="The images to visualize.")
    parser.add_argument('--meshvalues',
                        nargs='+',
                        type=str,
                        default=None,
                        help="Values to map on vertices.")
    parser.add_argument('--imglabels',
                        nargs='+',
                        type=str,
                        default=None,
                        help="Segmentation maps.")
    parser.add_argument('--clim',
                        type=float,
                        default=None,
                        nargs=2,
                        help="Color limits for value visualization.")
    parser.add_argument('--cpos',
                        type=str,
                        default=None,
                        help="A file containing a camera position, see cposes for examples")
    parser.add_argument('-s', '--screenshot',
                        type=str,
                        nargs='?',
                        const="/mnt/c/Users/Fabian/Desktop/",
                        default=None,
                        help="Optionally specify a dir where screenshots are stored")
    parser.add_argument('--label_mode',
                        type=str,
                        default='contour',
                        help="Either 'contour' or 'fill'")

    args = parser.parse_args()

    if args.meshes and not args.images:
        ### Show meshes only ###
        meshes = args.meshes
        values = args.meshvalues if args.meshvalues else [None] * len(args.meshes)

        # Broadcast values and meshes
        if len(values) == 1 and len(meshes) > 1:
            values = len(meshes) * values
        elif len(meshes) == 1 and len(values) > 1:
            meshes = len(values) * meshes
        elif len(meshes) != len(values):
            raise ValueError(f"Number of meshes ({len(meshes)}) and number of mesh value files should either be 1 or equal.")

        for i, (m, v) in enumerate(zip(meshes, values)):
            if v is not None:
                vertex_values = load_vertex_values(v)
                if vertex_values.ndim == 1:
                    vertex_values = np.expand_dims(vertex_values, 0)
            else:
                vertex_values = [None]
            for vv in vertex_values:
                vis_mesh(
                    load_mesh(m),
                    screenshot=os.path.join(args.screenshot, f"screenshot_{i}.png") if args.screenshot else None,
                    clim=args.clim,
                    vertex_values=vv,
                    title=": ".join([m, v]) if v else m,
                    cpos=np.load(args.cpos, allow_pickle=True).item() if args.cpos else None
                )

        return

    ### Show images ###
    if args.imglabels is not None and args.meshes is not None:
        raise ValueError("Please specify either --meshes or --imglabels")

    if args.meshes is None:
        if args.imglabels is None:
            args.imglabels = [None] * len(args.images)
        ### Show 2D slices of images potentially with voxel labels"""
        for img, label in zip(args.images, args.imglabels):
            vis_img_slices(
                img=load_img3D(img),
                label=load_img3D(label) if label is not None else None,
                output_file=args.screenshot,
                label_mode=args.label_mode,
                title=img
            )

        return

    # Show image and mesh label
    for img, label in zip(args.images, args.meshes):
        vis_img_slices(
            img=load_img3D(img),
            label=load_mesh(label),
            output_file=args.screenshot,
            label_mode=args.label_mode,
            title=img
        )

    return


if __name__ == "__main__":
    main()
