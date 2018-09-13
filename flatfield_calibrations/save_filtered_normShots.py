
# save the shots into train and test, save the 
import h5py
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import argparse
import numpy as np

import sys

from loki.RingData.DiffCorr import DiffCorr
from cali_utils import *


parser = argparse.ArgumentParser(description='Compute difference correlation by pairing single intensity correlations.')
parser.add_argument('-r','--run', type=int,
                   help='run number')
parser.add_argument('-t','--samp_type', type=int,
                   help='type of data/n \
# Sample IDs\n\
# -1: Silver Behenate smaller angle\n\
# -2: Silver Behenate wider angle\n\
# 0: GDP buffer\n\
# 1: ALF BUffer\n\
# 2: GDP protein\n\
# 3: ALF protein\n\
# 4: Water \n\
# 5: Helium\n\
# 6: 3-to-1 Recovered GDP')

parser.add_argument('-q','--qmin', type=int,
                   help='index of minimum q used for pairing or the only q used for pairing')

parser.add_argument('-u','--qmax', type=int, default=None,
                   help='index of max q used for pairing or None')

parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/results/combined_tables/finer_q',
                   help='where to look for the polar data')


parser.add_argument('-f','--n_files', type=int, default=1,
                   help='number of groups/files to split the data into. this splits larger runs for the PCA')



def sample_type(x):
    return {-1:'AgB_sml',
    -2:'AgB_wid',
     0:'GDP_buf',
     1:'ALF_buf',
     2:'GDP_pro',
     3:'ALF_pro',
     4:'h2o',
     5:'he',
     6:'3to1_rec_GDP_pro'}[x]


args = parser.parse_args()


outlier_threshold = 2 # sigma from median is conidered an outlier, and filtered out


run_num = args.run

if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file

data_dir = args.data_dir
save_dir = args.out_dir

save_dir = os.path.join( args.out_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print save_dir


run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')

# output file to save data
out_file = [run_file.replace('.tbl','_normShots_%d.h5'%ii) for ii in range(args.n_files)]
    


if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    print("there is no mask stored with the shots")
    sys.exit()
    # mask = np.load('/reg/d/psdm/cxi/cxilp6715/results/shared_files/binned_pmask_basic.npy')


PI = f['polar_imgs']
# filter by photon energy. If the photon energy of the shot if not within 100 EV of the average, do not use
photon_energy=np.nan_to_num(f['ebeam']['photon_energy'].value)
mean_E=photon_energy.mean()
E_sigma=100.
shot_tage_to_keep=np.where( (photon_energy> (mean_E-E_sigma))\
    +(photon_energy< (mean_E-E_sigma)) )[0]

print('Num of shots to be used: %d'%(shot_tage_to_keep.size))

# figure which qs are used for pairing
qmin = args.qmin
qmax = args.qmax

if qmax is None:
    qcluster_inds = [qmin]
else:
    qcluster_inds = range(qmin,qmax+1) # qmax is included

# this script on does the clustering only



# normalize all the shots at each q index

for qidx in qcluster_inds:

    shots=PI[:,qidx,:][shot_tage_to_keep,None,:]
    this_mask = mask[qidx][None,:]

    all_norm_shots = norm_all_shots(shots.astype(np.float64),this_mask, outlier_threshold)
    # split data usig args.n_files
    split_size = int(all_norm_shots.shape[0]/args.n_files)
    for file_ind in range(args.n_files):
        print ("noramlizing shot set %d out of %d"%(file_ind+1, args.n_files))
        if args.n_files==1:
            norm_shots = all_norm_shots.copy()
            del all_norm_shots
        else:
            norm_shots=all_norm_shots[file_ind*split_size:(file_ind+1)*split_size]
        f_out = h5py.File(os.path.join(save_dir, out_file[file_ind]),'a')

        # divide into Train and test

        num_shots = norm_shots.shape[0]
        cutoff = int(num_shots*0.1) # use 10% of the shots as testing set
        partial_mask = this_mask.copy()

        ##### start looping through all the files
        q_group = 'q%d'%qidx
        if q_group not in f_out.keys():
            f_out.create_group(q_group)

        Train = norm_shots[cutoff:, partial_mask==1]
        Test = norm_shots[:cutoff, partial_mask==1]
        f_out.create_dataset('q%d/train_shots'%qidx,data=Train)
        f_out.create_dataset('q%d/test_shots'%qidx,data=Test)
f_out.close()
print("done!")