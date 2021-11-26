#!/usr/bin/env python3

"""
Create training ground truth mesh from 3D labeled data with
marching cubes algorithm
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import nibabel as nib

from argparse import ArgumentParser
from tqdm import tqdm

from utils.utils import create_mesh_from_file

ignore_files = ["._", ".DS_"]

def create_and_store_gt_meshes():
    parser = ArgumentParser(description="Create a mesh from input file(s)"\
                            " and store in target directory.")
    parser.add_argument("infiles",
                         metavar="INPUT",
                         nargs='+',
                         help="The input files or a directory")
    parser.add_argument("--outdir",
                        metavar="DIR",
                        dest="outdir",
                        default="meshes/",
                        help="An output directory. If not specified,"
                        " the output is stored in meshes/")

    args = parser.parse_args()
    output_dir = args.outdir

    if os.path.isdir(args.infiles[0]):
        input_dir = args.infiles[0]
        filenames = [os.path.join(input_dir, fn) for fn in\
                     os.listdir(input_dir)]
    else:
        filenames = args.infiles

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}.")

    for fn in tqdm(filenames, position=0, leave=True,\
                   desc="Processing files..."):
        name = os.path.basename(fn) # filename without path

        if any(x in name for x in ignore_files):
            continue # skip file

        create_mesh_from_file(fn, output_dir)

    print(f"Stored meshes in {output_dir}.")

if __name__ == "__main__":
    create_and_store_gt_meshes()
