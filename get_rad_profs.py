
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

def fit_line(data):
    """
    fit a line to the end points of a 1-D numpy array

    data, 1D numpy array, e.g. a radial profile
    """
    left_mean = data[:10].mean()
    right_mean = data[-10:].mean()
    slope = (right_mean - left_mean) / data.shape[0]
    
    x = np.arange(data.shape[0])
    return slope * x + left_mean

def norm_data(data):
    """
    Normliaze the numpy array 
     
    data, 1-D numpy array
    """
    data_min = data.min()
    data2 = data- data.min()
    return data2/ data2.max()


parser = argparse.ArgumentParser(description='Data grab: get radial Radial Profiles.\n\
    Use them to decide peak range for hit detection.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-m', '--maxcount', type=int, default=0, help='max shots to process')
parser.add_argument('-s', '--start', type=int, default=0, help='first event to process')
parser.add_argument('-f', '--fname', type=str, required=True, help='output basename')
parser.add_argument('-rmax', '--waxs_rmax', type=int, default=1100,
 help='maximum rmax in pixels for radial profiles')
parser.add_argument('-rmin', '--waxs_rmin', type=int, 
    default=100, help='minimum rmin in pixels for radial profiles')
parser.add_argument('-d', '--out_dir', type=str,default ='/reg/d/psdm/cxi/cxilp6715/scratch/polar_data',
help='dir in which to store the polar data output')
parser.add_argument('-mf', '--mask_file', type=str,
    required=True,
    help=".npy file containing mask for the polar data")



args = parser.parse_args()
    
#############parameters to change
# img_sh: n-by-m pixel shape of the cspad
# cent_fname: .npy file storing the center of the detector in pixel unit, use NP rings to find center
# ds_str: change to the correct experiment number

# run number passed as string
run = args.run
start = args.start

#~~~~~analysis  parameters BEGIN
#####may need to change this for nanofocus chamber
img_sh = (1734, 1731)

# point where forward beam intersects detector
###################
# Need to find new center
###################
cent_fname = '/reg/d/psdm/cxi/cxilp6715/results/shared_files/center.npy'
mask_fname = args.mask_file
cent = np.load( cent_fname)
mask = np.load( mask_fname) 

#~~~~~ WAXS parameters
# minimium and maximum radii for calculating radial profile
waxs_rmin = args.waxs_rmin # pixel units
waxs_rmax = args.waxs_rmax
rp = RadialProfile( cent, img_sh, mask=mask  )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~ data get parameters
#load the data events for the given run
##########################################
#Need to change the experiment number to cxilr67##
##########################################
ds_str = 'exp=cxilp6715:run=%d:smd' % run
ds = psana.MPIDataSource(ds_str)
events = ds.events()

# open the detector obj
# change cspad alias!
#############################################
cspad = psana.Detector('CxiDs1.0:Cspad.0')
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
# save some parameters used in interpolation
d = {'polar_params':{'rmin':waxs_rmin,'rmax':waxs_rmax,\
 'center':cent,'mask':np.array(mask,dtype=int)} }
smldata.save(d)

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

#   WAXS FOR HIT FINDING~~~~~~~~~~~~~~
    rad_pro = rp.calculate(img)[waxs_rmin:waxs_rmax]
    print rad_pro.shape
    smldata.event(radial_profs=rad_pro.astype(np.float32))
    count+=1
    print("Images processed: %d out of %d events..."%(count,i+1))

smldata.save()


