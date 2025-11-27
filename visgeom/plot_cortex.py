#
# Created on Tue Aug 29 2023
#
# Copyright (c) 2023 Fabian Bongratz
#
# Script that creates a more structured output of cortical surfaces, with all views at a time.
#
# Usage:
#     python plot_cortex.py --subject_dir <PATH-TO-SUBJECT-DIR> --output_dir <PATH-TO-OUTPUT-DIR>

import os
import sys
import argparse
import pyvista as pv
import numpy as np
import nibabel.freesurfer.io as fsio
import trimesh
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import normalize

from pathlib import Path
VISGEOM_PATH = Path(__file__).resolve().parent

cpos_dir = os.path.join(VISGEOM_PATH, 'cposes/study_cposes')

scalar_bar_params = {
    'vertical': True,
    'width': 0.2,
    'height': 0.8,
    'position_x': 0.5,
    'position_y': 0.1,
    'title_font_size': 20,
    'label_font_size': 20,
    'below_label': 'unknown',
}


def main():
    parser = argparse.ArgumentParser(description='Create output of cth study')
    parser.add_argument('--subject_dir', type=str, help='Subject directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory', required=True)
    parser.add_argument('--filetype', type=str, default='thickness', help="'thickness' or 'SCSR'")
    parser.add_argument('--no_legend', action='store_true', help='Do not show legend in output plots')
    parser.add_argument('--reference_regions', type=str, nargs='+', default=['absolute'],
                         help='Reference regions to use for relative thickness calculation. \n' \
                         'Options: absolute, global, postcentral. \n')
    parser.add_argument('--ico', type=int, default=7, help='Icosahedron subdivision level, default: 7')
    parser.add_argument('--autoclose', action='store_true', help='Automatically close the plots; useful if there are many.')
    parser.add_argument('--file_ending', type=str, default=None, help='Custom file ending. If not provided, will be figured out automatically.')
    args = parser.parse_args()
    sub = args.subject_dir

    if args.filetype == 'thickness':
        if args.file_ending is None:
            if args.ico == 7:
                file_ending = '.bi_thickness'
            else:
                file_ending = f'.bi_thickness_ico{args.ico}'
        else:
            file_ending = args.file_ending
        plot_params = {
            'cmap': 'RdYlGn',
            'smooth_shading': True,
            'specular': 0.0,
            'show_scalar_bar': False,
            'below_color': 'gray',
        }
    elif args.filetype == 'SCSR':
        if args.file_ending is None:
            file_ending = '.SCSR_z_corr0.8_perc0.95'
        else:
            file_ending = args.file_ending
        plot_params = {
            'cmap': 'jet_r',
            'smooth_shading': True,
            'specular': 0.0, 
            'show_scalar_bar': False,
            'below_color': 'gray',
        }
        # No reference regions for SCSR
        if args.reference_regions != ['absolute']:
            print("For SCSR only 'absolute' reference region is allowed. Setting reference region to 'absolute'.")
        args.reference_regions = ['absolute']
        # SCSR runs on ico5 surface
        args.ico = 5
    else:
        print("filetype must be 'thickness' or 'SCSR'")
        sys.exit(1)

    print('Reference regions: ', args.reference_regions)
    print('Icosahedron level: ', args.ico)

    if args.ico == 7:
        inflated_surf_file_lh = '/mnt/data/subjects/fsaverage/surf/lh.inflated'
        inflated_surf_file_rh = '/mnt/data/subjects/fsaverage/surf/rh.inflated'
    else:
        inflated_surf_file_lh = f'/mnt/data/subjects/fsaverage{args.ico}/surf/lh.inflated'
        inflated_surf_file_rh = f'/mnt/data/subjects/fsaverage{args.ico}/surf/rh.inflated'

    inflated_surf_lh = fsio.read_geometry(inflated_surf_file_lh)
    inflated_surf_rh = fsio.read_geometry(inflated_surf_file_rh)

    # Normalization for smoothing: currently done with inverse normalized distance
    lap_lh = trimesh.smoothing.laplacian_calculation(
        trimesh.Trimesh(inflated_surf_lh[0], inflated_surf_lh[1], process=False),
        equal_weight=False
    )
    lap_lh.setdiag(np.ones(lap_lh.shape[0]))
    lap_lh = lap_lh.tocsr()
    lap_lh = normalize(lap_lh, norm='l1', axis=1)
    lap_rh = trimesh.smoothing.laplacian_calculation(
        trimesh.Trimesh(inflated_surf_rh[0], inflated_surf_rh[1], process=False),
        equal_weight=False
    )
    lap_rh.setdiag(np.ones(lap_rh.shape[0]))
    lap_rh = lap_rh.tocsr()
    lap_rh = normalize(lap_rh, norm='l1', axis=1)

    print(lap_lh[0])
    print(lap_rh[0])

    if not os.path.exists(args.output_dir + '/histo'):
        os.makedirs(args.output_dir + '/histo')

    th_all = {}
    undef_mask = {}
    for hemi in ['lh', 'rh']:
        print(hemi)
        th_all[hemi] = {}
        # Load thickness
        thickness = fsio.read_morph_data(sub + '/surf/' + hemi + file_ending)
        # Load parcellation
        try:
            parc= fsio.read_annot(sub + '/label/' + hemi + '.aparc.DKTatlas40.annot', orig_ids=True)
        except FileNotFoundError:
            print(f"Individual mask not found, using fsaverage")
            parc = fsio.read_annot(f'/mnt/data/subjects/fsaverage{args.ico}/label/{hemi}.aparc.annot')

        undef_id = (0, -1)
        undef_mask[hemi] = np.isin(parc[0], undef_id)  # Save for later
        mask = ~undef_mask[hemi]
        ref_region_values = thickness[mask]
        # Save histogram without showing
        plt.hist(ref_region_values, bins=100)
        plt.savefig(args.output_dir + '/histo/' + os.path.basename(os.path.normpath(sub)) + '_' + hemi + '_hist_global.png')
        plt.clf()
        print(f'Number of vertices in the cortex w/o middle : {len(ref_region_values)}')
        ref_cth = np.mean(ref_region_values)
        print(f'Mean of all values: ', ref_cth)

        # Relative to global thickness excluding undefined region
        if 'global' in args.reference_regions:
            glob_cth = thickness / ref_cth
            th_all[hemi]['global'] = glob_cth

        # Relative to postcentral region
        if 'postcentral' in args.reference_regions:
            post_id = parc[1][parc[-1].index(b'postcentral')][-1]
            mask = parc[0] == post_id
            ref_region_values = thickness[mask]
            # Save histogram without showing
            plt.hist(ref_region_values, bins=100)
            plt.savefig(args.output_dir + '/histo/' + os.path.basename(os.path.normpath(sub)) + '_' + hemi + '_hist_postcentral.png')
            plt.clf()
            print(f'Number of vertices in postcentral region: {len(ref_region_values)}')
            ref_cth = np.mean(ref_region_values)
            print(f'Mean cth of postcentral region: ', ref_cth)
            post_cth = thickness / ref_cth
            th_all[hemi]['postcentral'] = post_cth

        # Absolute values
        th_all[hemi]['absolute'] = thickness

    # Prepare meshes
    lh_verts = inflated_surf_lh[0]
    rh_verts = inflated_surf_rh[0]
    lh_faces = np.hstack([np.ones([inflated_surf_lh[1].shape[0], 1], dtype=int) * inflated_surf_lh[1].shape[1], inflated_surf_lh[1]])
    rh_faces = np.hstack([np.ones([inflated_surf_rh[1].shape[0], 1], dtype=int) * inflated_surf_rh[1].shape[1], inflated_surf_rh[1]])
    lh_mesh = pv.PolyData(lh_verts, lh_faces)
    rh_mesh = pv.PolyData(rh_verts, rh_faces)

    # Iterate over reference regions
    for _, ref in enumerate(args.reference_regions):
    # for i_ref, ref in enumerate(['absolute']):
        th_lh = th_all['lh'][ref]
        th_rh = th_all['rh'][ref]
        # Iterate over smoothing
        # for smoothing in (0, 100):
        for smoothing in (50,):
            # Prepare values
            if smoothing > 0:
                for i in range(smoothing):
                    th_lh = lap_lh.dot(th_lh)
                    th_rh = lap_rh.dot(th_rh)
            if ref == 'absolute':
                if args.filetype == 'thickness':
                    clim = (1.01, 4)
                    lh_mesh['cth'] = np.clip(th_lh, *clim)
                    rh_mesh['cth'] = np.clip(th_rh, *clim)
                    # Set undefined region to gray
                    lh_mesh['cth'][undef_mask['lh']] = 0.99
                    rh_mesh['cth'][undef_mask['rh']] = 0.99
                    # Avoid visualization errors with gray region
                    clim = (1.0, 4.0)
                elif args.filetype == 'SCSR':
                    clim = (-1.99, 0.00)
                    lh_mesh['cth'] = np.clip(th_lh, *clim)
                    rh_mesh['cth'] = np.clip(th_rh, *clim)
                    # Set undefined region to gray
                    lh_mesh['cth'][undef_mask['lh']] = -2.01
                    rh_mesh['cth'][undef_mask['rh']] = -2.01
                    # Avoid visualization errors with gray region
                    clim = (-2.0, 0.0)
            else:
                # Relative thickness
                clim = (0.51, 1.0)
                lh_mesh['cth'] = np.clip(th_lh, *clim)
                rh_mesh['cth'] = np.clip(th_rh, *clim)
                # Set undefined region to gray
                lh_mesh['cth'][undef_mask['lh']] = 0.49
                rh_mesh['cth'][undef_mask['rh']] = 0.49
                # Avoid visualization errors with gray region
                clim = (0.5, 1.0)

            # Create moved copy of meshes for joint visualization
            lh_mesh_moved = deepcopy(lh_mesh)
            lh_mesh_moved.points[:, 0] -= 35
            rh_mesh_moved = deepcopy(rh_mesh)
            rh_mesh_moved.points[:, 0] += 35

            # Prepare plot
            if args.no_legend:
                p = pv.Plotter(shape=(2, 4), window_size=(2240, 1080))
            else:
                n_rows = 2
                n_cols = 4
                row_weights = [1 for i in range(n_rows)]
                col_weights = [0.6] + [1 for i in range(n_cols)]
                groups = [  (np.s_[:], 0),    (0, 1) ]
                p = pv.Plotter(shape=(n_rows, n_cols+1),
                               row_weights=row_weights,
                               col_weights=col_weights, groups=groups,
                               window_size=(2240, 500*n_rows))
                p.add_mesh(lh_mesh, opacity=0., clim=clim, **plot_params)
                # p = pv.Plotter(shape=(2, 5), window_size=(2240, 1080))
                # p.add_mesh(lh_mesh, opacity=0., **plot_params)
                if ref == 'absolute':
                    if args.filetype == 'thickness':
                        p.add_scalar_bar(title='cth (mm)', **scalar_bar_params)
                    elif args.filetype == 'SCSR':
                        p.add_scalar_bar(title='SCSR z-score', **scalar_bar_params)
                else:
                    p.add_scalar_bar(title=f'relative cth', **scalar_bar_params)
                # Uncomment this to show also the reference region in the scalar bar title
                # if ref == 'absolute':
                #     p.add_scalar_bar(title='cth (mm)', **scalar_bar_params)
                # else:
                #     p.add_scalar_bar(title=f'cth/mean({ref})', **scalar_bar_params)

            x = 0 if args.no_legend else 1
            for view in ['lat', 'med', 'sup', 'inf', 'ant', 'post']:
                if view in ('sup', 'inf', 'ant', 'post'):
                    # Show both hemispheres in pyvista plot
                    p.subplot(1, x - 4)
                    p.add_text(view, color='white', font_size=10, position='upper_edge')
                    p.add_mesh(lh_mesh_moved, clim=clim, **plot_params)
                    p.add_mesh(rh_mesh_moved, clim=clim, **plot_params)
                    cpos = np.load(os.path.join(cpos_dir, 'lh_' + view + '.npy'), allow_pickle=True)
                    p.camera_position = cpos.item()
                    x += 1
                else:
                    p.subplot(0, x)
                    p.add_text(view + '/lh', color='white', font_size=10, position='upper_edge')
                    p.add_mesh(lh_mesh, clim=clim, **plot_params)
                    cpos = np.load(os.path.join(cpos_dir, 'lh_' + view + '.npy'), allow_pickle=True)
                    p.camera_position = cpos.item()
                    x += 1

                    p.subplot(0, x)
                    p.add_text(view + '/rh', color='white', font_size=10, position='upper_edge')
                    p.add_mesh(rh_mesh, clim=clim, **plot_params)
                    cpos = np.load(os.path.join(cpos_dir, 'rh_' + view + '.npy'), allow_pickle=True)
                    p.camera_position = cpos.item()
                    x += 1

            ref_ = f'ref-{ref}' if ref != 'absolute' else ''
            ident = f'{ref_}-thickness' if args.filetype == 'thickness' else f'-SCSR'
            p.title = f"{os.path.basename(os.path.normpath(sub))}_{ident}_smooth-{smoothing}"
            out_dir_plot = args.output_dir + '/' + ref + '_smooth_' + str(smoothing)
            if not os.path.exists(out_dir_plot):
                os.makedirs(out_dir_plot)
            out_fn = out_dir_plot + '/' + p.title + '.png'
            if args.autoclose:
                pos = p.show(screenshot=out_fn, interactive=False, auto_close=True, return_cpos=True)
            else:
                pos = p.show(screenshot=out_fn, interactive=True, auto_close=False, return_cpos=True)
            # p.screenshot(out_fn, return_img=False)
            p.close()
    pv.close_all()

if __name__ == '__main__':
    main()
