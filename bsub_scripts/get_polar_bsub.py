import numpy as np
import os
import sys

# this scripts divide a run into nfiles number of jobs to submit for polar intensities interpolations
# each file has roughly the same number of raw scattering shots to process

run = sys.argv[1] # run number
max_evt = int(sys.argv[2]) # the maximum number of shots in the run. Look for this in data log
nfiles = int(sys.argv[3]) # number of files to divide this into
num_rbin = sys.argv[4] # number of radial pixels to end up with for the polar intensities
rmin = sys.argv[5] # minimum radius to start interpolation, pixel unit
rmax = sys.argv[6] # maxmimum radiua for interpolation, pixel unit

evt_list = np.array_split(np.arange( max_evt), nfiles )
# change this to your directory
program ="python /reg/neh/home5/qiaoshen/GitHub/lcls_run16_analysis_scripts/get_polar_data.py"
#######################################################################################3

for i,evts in enumerate(evt_list):
    evt_start = evts[0]
    n_evts = len( evts)
    # change this to where you want the bsub logs to be stored
    ############################################################
    logfile = "/reg/neh/home/qiaoshen/logs/run%s_%d.log"%(run,i)
    ############################################################
    
    fname = "run%s_%d.h5"%(run,i)
    # change this to directory where you want to store the polar data output
    ######################################################################
    outdir = '/reg/d/psdm/cxi/cxilp6715/scratch/polar_data/higher_q_450_800'
    ######################################################################
    if os.path.exists( os.path.join(outdir, fname) ):
        print("Skipping %s"%fname)
        continue
    cmd = ["bsub",
        "-o %s"%logfile,
        "-q psanaq",
        program,  
        "-r %s"%run,
        "-s %d"%evt_start,
        "-f %s"%fname,
        "-m %d"%n_evts,
        "-b %s"%num_rbin,
        "-rmax %s"%rmax,
        "-rmin %s"%rmin,
        "-d %s"%outdir]


    cmd = " ".join( cmd)

    os.system(cmd)
# print cmd
