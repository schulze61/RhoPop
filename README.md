[![DOI](https://zenodo.org/badge/703276913.svg)](https://zenodo.org/doi/10.5281/zenodo.10368147)


RhoPop is a software for the identification of compositionally distinct populations of small planets ($\lesssim 2 R_\oplus$). It employs mixture models in a hierarchical framework and the dynesty nested sampler for parameter and evidence estimates. RhoPop includes a density-mass grid of water-rich compositions from water mass fraction (WMF) 0-1.0 and a grid of volatile-free rocky compositions over a core mass fraction (CMF) range of 0.006-0.95. Both grids were calculated using the ExoPlex mass-radius-composition calculator (https://github.com/CaymanUnterborn/ExoPlex).

RhoPop is python 3 based and requires the following packages:
- numpy
- scipy
- maplotlib
- pandas
- dynesty (https://dynesty.readthedocs.io/en/stable/)

Once the required python packages are installed, RhoPop is intended to be 'plug-n-play', requiring only a mass-radius csv file to run. You will need the following columns:
 - `Planet': planet name
 - 'mass': planet mass in Earth masses
 - 'mass_err: - planet mass uncertainty in Earth masses
 - 'radius': planet radius in Earth radii
 - 'radius_err': planet radius uncertainty in Earth radii

 Save the M-R file in the MR_files directory. There are pre-loaded examples of data files you can use for reference. Change the dfile_root input to your M-R file name in the single_population.py, two_population.py, or three_population.py files.
 Below is a flow-chart of the intended workflow of RhoPop. We show schematic 1-D scaled density profiles for simplicity, but, we note, that the problem is inherently 2-D owing to the self-compression of planetary materials.

![rhopop_workflow](https://github.com/schulze61/RhoPop/assets/43186618/e1b03d7b-2727-4643-912d-94d1a18d0305)
