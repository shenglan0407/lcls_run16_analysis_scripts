import h5py
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Gather statistics on total intensities of shots.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-d', '--out_dir', type=str, required=True,
help='dir in which to store the output')
parser.add_argument('-e', '--delta', type=float, default=2., help='delta between the bin edges for binning intensities')
parser.add_argument('-m', '--min_int', type=float, default=0.0, help='minimum intensity to include in histogram')

args = parser.parse_args()

run=args.run
out_dir = args.out_dir

f = h5py.File('/reg/d/psdm/cxi/cxilr6716/scratch/flatfield_calibration/flat_det_imgs/fullImgs_run%d.h5'%run,'r')
f_out = h5py.File(os.path.join(out_dir,'run%d_ave_int.h5'%run),'w')

flattened_shots = f['flat_img']

# histogram the toalt mean intensities
total_intensity = f['ave_tot_int'].value

delta=args.delta
bins = np.arange(args.min_int,total_intensity.max()+delta,delta) #use universal bins

labels = np.digitize(total_intensity,bins)
unique_bins=np.unique(labels[labels>0])
bin_centers_with_shots =  bins[unique_bins]-delta/2.
num_shots_per_bin = np.array([np.sum(labels==ll) for ll in unique_bins])

#gather average shots and statistics
ave_shots = np.zeros( (unique_bins.size, flattened_shots.shape[-1]))
std_shots = np.zeros_like(ave_shots)
print("Total number of bins: %d"%unique_bins.size)
for idx, ii in enumerate(unique_bins):
    inds=np.where(labels==ii)[0]
    inds = list(inds)
    print ("Bin %d; num shots: %d"%(ii, len(inds)) )
    shots=flattened_shots[inds]
    std_shots[idx] = shots.std(0)/ np.sqrt(float(len(inds)))
    ave_shots[idx] = shots.mean(0)

f_out.create_dataset('ave_flat_shots', data=ave_shots)
f_out.create_dataset('ave_flat_shots_err',data=std_shots)

f_out.create_dataset('bin_centers',data = bin_centers_with_shots)
f_out.create_dataset('num_shots_per_bin',data=num_shots_per_bin)
f_out.create_dataset('bin_delta',data=delta)

f_out.close()
f.close()
print ("done!")
