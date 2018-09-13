import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

import argparse

from numpy import polyval, polyfit


parser = argparse.ArgumentParser(description='Gather statistics on total intensities of shots.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-d', '--out_dir', type=str, required=True,
help='dir in which to store the output')
parser.add_argument('-n', '--name', type=str, required=True, 
    help='name of the file')

parser.add_argument('-s', '--start', type=int, required=True, 
    help='start index of pixels to calibrate')
parser.add_argument('-e', '--end', type=int, required=True, 
    help='end index of pixels to calibrate')

parser.add_argument('-x', '--degree', type=int, required=True, 
    help='polynomial degree')
args = parser.parse_args()

run = args.run
out_dir = args.out_dir
name = args.name

fname=os.path.join(out_dir,name)
f_out = h5py.File(fname,'w')

f = h5py.File('/reg/d/psdm/cxi/cxilr6716/results/flatfield_calibration/ave_int_statisitcs/run%d_ave_int.h5'%run,
    'r')

flat_ave_shots = f['ave_flat_shots'].value
# ave_bin_int = f['bin_centers'].value
ave_bin_int = flat_ave_shots.mean(-1)
num_shots = f['num_shots_per_bin'].value

# thresholding
select=num_shots>10
flat_ave_shots = flat_ave_shots[select]
ave_bin_int = ave_bin_int[select]

ref_int = ave_bin_int[int(ave_bin_int.size/2)]

# fit to polynomial
degree = args.degree
start= args.start
end=args.end
poly_coefs = np.zeros( (end-start,degree+1) )

for idx,ii in enumerate(range(start,end)):
    y = flat_ave_shots[:,ii]
    cc = polyfit(ave_bin_int-ref_int, y, degree)
    poly_coefs[idx] = cc

# save polynomial poly_coefs
f_out.create_dataset('poly_coefs', data = poly_coefs)
f_out.create_dataset('ave_bin_int', data=ave_bin_int)
f_out.create_dataset('num_shots_per_bin', data=num_shots)
f_out.create_dataset('ref_shot_ind', data=int(ave_bin_int.size/2))
f_out.close()
print("done!")
