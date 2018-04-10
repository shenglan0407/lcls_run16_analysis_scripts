import numpy as np
import os
import sys

# this scripts divide a run into nfiles number of jobs to submit for polar intensities interpolations
# each file has roughly the same number of raw scattering shots to process

run = sys.argv[1] # run number
max_evt = int(sys.argv[2]) # the maximum number of shots in the run. Look for this in data log
nfiles = int(sys.argv[3]) # number of files to divide this into
rmin = sys.argv[4] # minimum radius to start interpolation, pixel unit
rmax = sys.argv[5] # maxmimum radiua for interpolation, pixel unit

evt_list = np.array_split(np.arange( max_evt), nfiles )
# change this to your directory
program ="python /reg/neh/home5/qiaoshen/GitHub/lcls_run16_analysis_scripts/get_rad_profs.py"
# change the mask file to the one for LR67
mask_file='/reg/d/psdm/cxi/cxilp6715/results/shared_files/mask_rough4.npy'
#######################################################################################

for i,evts in enumerate(evt_list):
    evt_start = evts[0]
    n_evts = len( evts)
    # change this to where you want the bsub logs to be stored
    ############################################################
    logfile = "/reg/neh/home/qiaoshen/logs/rad_profs_run%s_%d.log"%(run,i)
    ############################################################
    
    fname = "rad_profs_run%s_%d.h5"%(run,i)
    # change this to directory where you want to store the polar data output
    ######################################################################
    outdir = '/reg/d/psdm/cxi/cxilp6715/scratch/rad_prof_test_run16'
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
        "-rmax %s"%rmax,
        "-rmin %s"%rmin,
        "-d %s"%outdir,
        "-mf %s"%mask_file]


    cmd = " ".join( cmd)

    # os.system(cmd)

    print cmd
    break
