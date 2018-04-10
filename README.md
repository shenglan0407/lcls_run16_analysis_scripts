# lcls_run16_analysis_scripts
for real time analysis

## Install loki in a custom conda env on a psana node
Data analysis tools are in this repo: https://github.com/dermen/loki

This repo is very much alive. Please pull and reinstall often if you want to use it!

Installation:

Log into pslogin node

Source conda env: source /reg/g/psdm/etc/psconda.sh

Make your own conda environment by cloning an available one: 
conda create --name my_ana1.3.50 --clone ana-1.3.50

*conda info --envs (shows all available conda envs)

Activate you copied env: conda activate my_ana1.3.50

Cd into loki directory and do: python setup.py install

Now you should be able to import loki and it’s functionalities in your conda env.

## Analysis Workflow
1. Make a rough mask to mask out detector gaps: make_a_single_mask.py (-h flag for usage)

2. Determine the center of the detector where the x-ray beam crosses the detector plane

3. Run get_rad_prof.py for some runs to find the location of the water peak in pixel units. Use bsub_scripts/get_rad_profs_bsub.py to submit multiple jobs per run. Open bsub_scripts/get_rad_profs_bsub.py in text editor to see usage.

4. Open get_polar_data.py to edit detetor parameters, including center position, and the range in which to find the water peak for hit detection.

5. Run get_polar_data.py to interpolate intensities to polar coordinates for runs that are finished. Use bsub_scripts/get_polar_bsub.py to submit multiple jobs per run. Open bsub_scripts/get_polar_bsub.py in text editor to see usage.

6. Run combine_files.py to consolidate polar data from each run

7. Run pca_denoise_numComp_diagnostics.py to get preliminary correlations after removal of eigenimages, upto num_pca number of eigneimages.
