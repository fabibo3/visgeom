# visgeom
Scripts for convenient visualization of 3D meshes and associated per-vertex values from the command line based on [pyvista](https://pyvista.org/). Basic image visualization can also be done. It can deal with FreeSurfer file types, .plys, and numpy arrays in arbitrary combinations.

## Usage
Download this repository
```
git clone git@github.com:fabibo3/visgeom.git
```
Install conda environment and activate
```
cd visgeom
conda env create -f requirements.yml
conda activate visgeom
```
Install the `visgeom` package in editable mode
```
pip install -e .
```
Visualize meshes, e.g., for a brain with cortical thickness
```
visgeom --meshes /path/to/lh.inflated --meshvalues /path/to/lh.thickness.npy --clim 1 4
```
When given multiple meshes or meshvalues, they are visualized sequentially.
