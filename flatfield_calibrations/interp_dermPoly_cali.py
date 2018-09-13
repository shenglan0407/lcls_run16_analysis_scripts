import h5py

import numpy as np
import h5py
from scipy.signal import argrelmax

from loki.RingData import RadialProfile, InterpSimple
from loki.utils.postproc_helper import smooth, bin_ndarray

import os

import sys
run = int(sys.argv[1])
interp_rmin=int(sys.argv[2])
interp_rmax=int(sys.argv[3])

def unflatten_shots(flat_shots,mask):
    num_shots = flat_shots.shape[0]
    shape = mask.shape
    flat_mask = mask.reshape(shape[0]*shape[1]
                            )
    shots = np.zeros( (num_shots,shape[0]*shape[1]), dtype = flat_shots.dtype)
    shots[:, flat_mask ] = flat_shots
    
    return shots.reshape( (num_shots,shape[0],shape[1]))



mask=np.load('/reg/d/psdm/cxi/cxilr6716/results/masks/basic_psana_mask.npy')

img_sh = mask.shape

# point where forward beam intersects detector
cent_fname = '/reg/d/psdm/cxi/cxilp6715/results/shared_files/center.npy'
cent = np.load( cent_fname)

#~~~~~ WAXS parameters
# minimium and maximum radii for calculating radial profile
waxs_rmin = 100 # pixel units
waxs_rmax = 1110
rp = RadialProfile( cent, img_sh, mask=mask  )

#interp params
phibins=360

rbin_fct = 1.
# adjust so our edge is a multiple of rbin factor
interp_rmax = int( interp_rmin + np.ceil( (interp_rmax - interp_rmin) / rbin_fct)*rbin_fct )
nphi = int( 2 * np.pi * interp_rmax )
if phibins>0:
    phibin_fct = np.ceil( nphi / float( phibins ) )
else:
    phibin_fct=1
nphi = int( np.ceil( 2 * np.pi * interp_rmax/phibin_fct)*phibin_fct) # number of azimuthal samples per bin

Interp = InterpSimple( cent[0], cent[1] , interp_rmax, interp_rmin, nphi, img_sh)  
pmask = Interp.nearest(mask).astype(int).astype(float)
print pmask.shape


f_out=h5py.File('/reg/d/psdm/cxi/cxilr6716/scratch/flatfield_calibration/interp_dermPoly_shots/interp_%d_%d/run%d_interp_dermPoly_shots.h5'%(interp_rmin,interp_rmax,run),
    'w')
        

# interpolate the uncalibrated shots
print ('interpolating original shots...')
single_int_dir = '/reg/d/psdm/cxi/cxilr6716/scratch/flatfield_calibration/derm_poly_imgs'
f_imgs = h5py.File(os.path.join(single_int_dir,'cali_run%d.h5'%run),'r')

single_shots = f_imgs['corrected_img'][:]

interp_images = np.zeros((single_shots.shape[0],pmask.shape[0],pmask.shape[1]) )
radial= np.zeros( (single_shots.shape[0],waxs_rmax-waxs_rmin) )
for idx,img in enumerate(single_shots):
    interp_images[idx] = Interp.nearest( img) * pmask
    radial[idx] = rp.calculate(img)[waxs_rmin:waxs_rmax]
    
f_out.create_dataset('single_shots_rp',data=radial)
f_out.create_dataset('single_shots_interp',data=interp_images)


f_out.close()
print "done"
    # save