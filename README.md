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
