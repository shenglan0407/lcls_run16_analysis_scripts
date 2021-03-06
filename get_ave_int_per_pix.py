
import sys
import os

import numpy as np
import h5py
import argparse
from scipy.signal import argrelmax
import pylab as plt

from loki.RingData import RadialProfile, InterpSimple
from loki.utils.postproc_helper import smooth, bin_ndarray
import psana


parser = argparse.ArgumentParser(description='Data grab: get radial Radial Profiles.\n\
    Use them to decide peak range for hit detection.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-m', '--maxcount', type=int, default=0, help='max shots to process')
parser.add_argument('-s', '--start', type=int, default=0, help='first event to process')
parser.add_argument('-f', '--fname', type=str, required=True, help='output basename')
parser.add_argument('-d', '--out_dir', type=str,default ='/reg/d/psdm/cxi/cxilp6715/scratch/polar_data',
help='dir in which to store the polar data output')

args = parser.parse_args()
    
#############parameters to change
# img_sh: n-by-m pixel shape of the cspad
# cent_fname: .npy file storing the center of the detector in pixel unit, use NP rings to find center
# ds_str: change to the correct experiment number

# run number passed as string
run = args.run
start = args.start

#~~~~~analysis  parameters BEGIN

#~~~ data get parameters
#load the data events for the given run
##########################################
#Need to change the experiment number to cxilr67##
##########################################
ds_str = 'exp=cxilr6716:run=%d:smd' % run
ds = psana.MPIDataSource(ds_str)
events = ds.events()

# open the detector obj
# change cspad alias!
#############################################
cspad = psana.Detector('CxiDs2.0:Cspad.0')
#############################################

# make some output dir
outdir = args.out_dir
if not os.path.exists(outdir):
    os.makedirs(outdir)

# make output file
prefix = args.fname
out_fname = os.path.join( outdir, prefix)

#small dat
smldata = ds.small_data(out_fname)

#maks file
basic_mask = np.load('/reg/d/psdm/cxi/cxilr6716/results/masks/basic_psana_mask.npy')

count = 0
seen_evts = 0

for i,evt in enumerate(events):
#   keep this first, i should never be < 0
    if i < start:
        #print ("skipping event %d/%d"%(i+1, start))
        continue

    img = cspad.image(evt)
    
    seen_evts += 1
    if seen_evts == args.maxcount:
        break
    
    if img is None:
        print("img is None")
        continue

#   total intensities~~~~~~~~~~~~~~
    total_intensities = img[basic_mask].mean().astype(np.float32)
    smldata.event(ave_ints=total_intensities)
    count+=1
    print("Images processed: %d out of %d events..."%(count,i+1))

smldata.save()


