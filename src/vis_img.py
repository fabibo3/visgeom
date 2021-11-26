#!/usr/bin/env python3

""" Visualization of raw images """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

from argparse import ArgumentParser
from utils.visualization import show_img_slices_3D

def vis_img3D():
    parser = ArgumentParser(description="Visualize 3D image data.")
    parser.add_argument('filenames',
                        nargs='+',
                        type=str,
                        help="The filenames or the name of one folder to visualize.")
    parser.add_argument('--nolabel',
                        dest='show_label',
                        action='store_false',
                        help="Disable visualization of ground truth labels.")
    parser.add_argument('--label',
                        dest='label_mode',
                        type=str,
                        default='contour',
                        help="Either 'contour' or 'fill'.")
    parser.add_argument('--dataset',
                        dest='dataset',
                        default="Cortex",
                        type=str,
                        help="Either 'Cortex' or 'Hippocampus'")
    parser.add_argument('--labels_from_mesh',
                        dest='labels_from_mesh',
                        type=str,
                        nargs='+',
                        default=None,
                        help="Use a voxelized mesh (given its path) as label"
                        " in the image visualization. If not specified, it is"
                        " searched for segmentation labels of nifity format.")
    parser.add_argument('--output',
                        dest='output_file',
                        type=str,
                        default=None,
                        help="Store the output in a file.")

    args = parser.parse_args()
    if os.path.isdir(args.filenames[0]):
        filenames = args.filenames[0]
    else:
        filenames = args.filenames
    show_img_slices_3D(filenames, args.show_label, args.dataset,
                       args.label_mode, args.labels_from_mesh, args.output_file)

if __name__ == "__main__":
    vis_img3D()
