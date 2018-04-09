from psana import *
import numpy as np
import h5py
import sys
import scipy
import os
import argparse


def get_imgs(evets):
    imgs=[]
    for nevent,evt in enumerate(events):
        if nevent>=1000: 
            break
        try:
            img = det.image(evt)
        except ValueError:
            return None
        imgs.append(img)
    imgs=np.array(imgs)

    return imgs

parser = argparse.ArgumentParser(description='Make a mask for one run. mask file save as .npy file')
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-d', '--dir', type=str, required=True, help='directory in which to save the mask')
parser.add_argument('-n', '--name', type=str, default='rough_mask.npy', help='name of the mask File')


args = parser.parse_args()
run=args.run
out_dir=args.dir
name=args.name
save_name=os.path.join(out_dir,name)

if os.path.isfile(save_name):
    print('Will not overwrite mask file that already exists')
    print('did not make new mask file %s'%save_name)

    sys.exit()

# make mask for this run 
#################################################
# change experiment name to cxilr67##
ds_str = 'exp=cxilp6715:run=%d:smd' % run
#################################################
try:
    ds = MPIDataSource(ds_str)
except RuntimeError:
    print("run %d does not exit"%run)
    
print('making masks for run %d'%run)
print('making detector mask...')
det = Detector('CxiDs1.0:Cspad.0')
det_mask = det.mask(run,calib=True,status=True,edges=True,central=True,unbond=True,unbondnbrs=True)
det_mask = det.image(run,det_mask).astype(bool)

events = ds.events()
# now get the negative pixel mask from gathering a few hundred images
print('making negative pixel mask...')
imgs=get_imgs(events)
if imgs is None:
    print("run %d does not exit. Did not make mask!"%run)
    sys.exit()
im = np.median(imgs,0)
negative_pixel_mask=im>0
negative_pixel_mask = ~(scipy.ndimage.morphology.binary_dilation(~negative_pixel_mask, iterations=1) )

print('saving full mask in %s'%save_name)
mask= det_mask*negative_pixel_mask
np.save(save_name,mask)
print('Done!')  