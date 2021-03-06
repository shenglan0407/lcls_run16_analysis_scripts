
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
from nonLinCorr import *
from cali_utils import *

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
parser.add_argument('-b', '--rbins', type=int, default=35, help='number of radial bins')
parser.add_argument('-p', '--phibins', type=int, default=360, help='number of phi bins, if 0 then do not interpolate in phi axis')
parser.add_argument('-rmax', '--interp_rmax', type=int, default=450, help='maximum rmax in pixels to interpolate')
parser.add_argument('-rmin', '--interp_rmin', type=int, default=100, help='minimum rmin in pixels to interpolate')
parser.add_argument('-d', '--out_dir', type=str,default ='/reg/d/psdm/cxi/cxilr6716/scratch/polar_data',
help='dir in which to store the polar data output')
parser.add_argument('-c', '--calib', type=int,required=True,
    help='if True, apply flatfield calibration')

parser.add_argument('-w', '--waterpeak_filter', type=int,default =1,
    help='if > 0 then filter for water peaks. use when solution scattering')


parser.add_argument('-k', '--correct_polarization', type=int,default =1,
    help='if > 0 then correct for polarization')



args = parser.parse_args()
do_calib = bool(args.calib) 

# run number passed as string
run = args.run
start = args.start

#~~~~~analysis  parameters BEGIN

img_sh = (1738, 1742)

# load calibration coefs
f_coefs=h5py.File('/reg/d/psdm/cxi/cxilr6716/results/flatfield_calibration/copper_cali_coefs.h5','r')
cn = f_coefs['coefs_10'].value

# load polarization correction
correct_polarization = args.correct_polarization>0
polar_correct = np.load('/reg/d/psdm/cxi/cxilr6716/scratch/derm/polarization_-400encode.npy')
# point where forward beam intersects detector
cent_fname =  "/reg/data/ana04/cxi/cxilr6716/scratch/derm/center_prelim_bigger.npy"
# mask_fname = '/reg/d/psdm/cxi/cxilr6716/results/masks/run%d_masks.h5'%run
# if not os.path.isfile(mask_fname):
#     print('mask file does not exist for run %d'%run)
#     sys.exit()
    
cent = np.load( cent_fname)
# f_mask= h5py.File(mask_fname,'r')
#####
###Also have a common mask for calibration purpose
basic_mask = np.load('/reg/d/psdm/cxi/cxilr6716/results/masks/basic_psana_mask.npy')
# use basic mask for now
mask = np.load('/reg/d/psdm/cxi/cxilr6716/results/masks/basic_psana_mask.npy')
# mask = f_mask['mask'].value
####

#~~~~~ WAXS parameters
# minimium and maximum radii for calculating radial profile
waxs_rmin = 100 # pixel units
waxs_rmax = 1110
rp = RadialProfile( cent, img_sh, mask=mask  )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~ hit findin' parameters... 
waterpeak_filter=args.waterpeak_filter>0
beta = 50 # smoothing factor
window_size = 200 # pixel units

# maxima detection
order = 250 # defines minimum neighborhood for local maxima (in radial pixel units)
# paraemters for peak validation
pk_range = (604, 804) # radial pixel units, relative to the range of the radial profiles
# e.g. will ensure the detected peak lies on rad_pofile[ pk_range[0] : pk_range[1]] 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~ interp parameters

# min ring radii
# desired dimension of image, these will be approximate
interpolate_PI=True
rbins =args.rbins
phibins = args.phibins

interp_rmin = args.interp_rmin
interp_rmax = args.interp_rmax

rbin_fct = np.floor( (interp_rmax - interp_rmin) / rbins)
# adjust so our edge is a multiple of rbin factor
interp_rmax = int( interp_rmin + np.ceil( (interp_rmax - interp_rmin) / rbin_fct)*rbin_fct )
nphi = int( 2 * np.pi * interp_rmax )

if phibins>0:
    phibin_fct = np.ceil( nphi / float( phibins ) )
else:
    phibin_fct=1
nphi = int( np.ceil( 2 * np.pi * interp_rmax/phibin_fct)*phibin_fct) # number of azimuthal samples per bin

rbin_new = (interp_rmax- interp_rmin ) / rbin_fct
phibin_new = nphi / phibin_fct 
binned_pol_img_sh = ( int(rbin_new), int(phibin_new) )
print("polar image dimensions:  %d x %d"%(rbin_new, phibin_new))

Interp = InterpSimple( cent[0], cent[1] , interp_rmax, interp_rmin, nphi, img_sh)  
pmask = Interp.nearest(mask).astype(int).astype(float)
pmask_bn = bin_ndarray( pmask, binned_pol_img_sh)

# print pmask.shape,pmask_bn.shape
# if (rbin_fct==1 and phibin_fct==1):
#     print('not binning polar img')
# sys.exit()
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
# save some parameters used in interpolation
d = {'polar_params':{'rmin':interp_rmin,'rmax':interp_rmax,\
 'center':cent,'mask':np.array(mask,dtype=int), 'pk_range':pk_range} }
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

    ##############################
    #Get the det image and calibrate it with coefficeints
    # Correct for polarization
    if correct_polarization:
        img/=polar_correct
    ave_img_int = (img[basic_mask]).mean()
    if (ave_img_int<1.3) or (ave_img_int>95.7):
        print('ave intensity of shot out of calibration range, skip!')
        continue

    if do_calib:
        # Get masked average intensity (use basic common mask)

        # Compute gain from coefs store
        c = lambda(i): polyVal(cn,i)
        # Unflatten gain to image size with mask
        gain = unflatten_shots( c(ave_img_int), basic_mask)[0]
        # Apply gain to image
        img*=gain

    ##############################
#   WAXS FOR HIT FINDING~~~~~~~~~~~~~~
    rad_pro = rp.calculate(img)[waxs_rmin:waxs_rmax]
    
    # if this is not a helium run, do the following
    if waterpeak_filter:
#       before peak fitting we smooth
        #flat_pro = smooth(rad_pro-fit_line(rad_pro), beta=beta, window_size=window_size) 
        smooth_pro =  smooth(rad_pro, beta=beta, window_size=window_size) 
        #norm_pro = norm_data(flat_pro)

#       we can find local maxima in the smoothed normalized radial profiles... 
        mx = argrelmax(smooth_pro, order=order)[0] 

#       make sure there is only one peak!
        if not len(mx) == 1:
            print("found too many maxs")
            continue

#       make sure the peak lies in the desired range.. 
        pk_pos = mx[0]
        if not pk_range[0] < pk_pos < pk_range[1] : 
            print("max peak is outside of range")
            continue

#       make sure the peak value is max in the original profile, 
#       because it was selected using line-subtracted profile
        pk_val = smooth_pro[ pk_pos] 
        if not smooth_pro[pk_range[0]] < pk_val and not smooth_pro[pk_range[1]] < pk_val :
            print("max is not a true max")
            continue  
#   if made it this far it is a hit, or run 96 helium only.. 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~Interpolation to polar
    if interpolate_PI:
        polar_img = Interp.nearest( img) * pmask

        if (rbin_fct==1 and phibin_fct==1):
            print('not binning polar img')
            smldata.event(polar_imgs=polar_img.astype(np.float32))
        else:
            polar_img_bn = bin_ndarray( polar_img, binned_pol_img_sh)* pmask_bn
            smldata.event(polar_imgs=polar_img_bn.astype(np.float32))
    smldata.event(radial_profs=rad_pro.astype(np.float32))
    count+=1
    print("Images processed: %d out of %d events..."%(count,i+1))
if interpolate_PI:
    smldata.save(polar_mask_binned=pmask_bn.astype(int))
    smldata.save(polar_mask=pmask.astype(int))
smldata.save()


