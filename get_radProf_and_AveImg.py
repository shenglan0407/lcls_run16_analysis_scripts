
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


parser = argparse.ArgumentParser(description='Data grab')
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-m', '--maxcount', type=int, default=0, help='max shots to process')
parser.add_argument('-s', '--start', type=int, default=0, help='first event to process')
parser.add_argument('-f', '--fname', type=str, default=None, help='output basename')

parser.add_argument('-d', '--out_dir', type=str,default ='/reg/d/psdm/cxi/cxilr6716/scratch/polar_data',
help='dir in which to store the polar data output')



args = parser.parse_args()
    

# run number passed as string
run = args.run
start = args.start

#~~~~~analysis  parameters BEGIN

img_sh = (1738, 1742)

# point where forward beam intersects detector
cent_fname =  "/reg/data/ana04/cxi/cxilr6716/scratch/derm/center_prelim_bigger.npy"
mask_fname = '/reg/d/psdm/cxi/cxilr6716/results/masks/basic_psana_mask.npy'
if not os.path.isfile(mask_fname):
    print('mask file does not exist for run %d'%run)
    sys.exit()
    
cent = np.load( cent_fname)
mask = np.load(mask_fname)

#~~~~~ WAXS parameters
# minimium and maximum radii for calculating radial profile
waxs_rmin = 40 # pixel units
waxs_rmax = 1140
rp = RadialProfile( cent, img_sh, mask=mask  )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~ data get parameters
#load the data events for the given run
ds_str = 'exp=cxilr6716:run=%d:smd' % run
ds = psana.MPIDataSource(ds_str)
events = ds.events()

# open the detector obj
cspad = psana.Detector('DsaCsPad')

# make some output dir
outdir = args.out_dir
if not os.path.exists(outdir):
    os.makedirs(outdir)

# make output file
if args.fname is None:
    prefix = 'BigAzzData-run%d-maxEvent%d-start%d.hdf5'%(run,args.maxcount, args.start)
else:
    prefix = args.fname
out_fname = os.path.join( outdir, prefix)

#small dat
smldata = ds.small_data(out_fname)


count = 0
seen_evts = 0

total_img =None

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
    
    if total_img is None:
        total_img=img
    else:
        total_img+=img

    smldata.event(radial_profs=rad_pro.astype(np.float32))
    count+=1
    print("Images processed: %d out of %d events..."%(count,i+1))

smldata.save(ave_img=total_img/float(count))
smldata.save()


