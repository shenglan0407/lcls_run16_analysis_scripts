import os
import h5py
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Combining data from serparat h5 files to one big one')
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-m', '--run_mx', type=int, default=None, help='if not none, understood that use a range of runs')
parser.add_argument('-d', '--data_dir', type=str, required = True, help='where the data is')
parser.add_argument('-s', '--save_dir', type=str, required = True, help='where to save the data')
args = parser.parse_args()

# define run numbers
if args.run_mx is None:
    runs = [args.run]
else:
    runs = range( args.run, args.run_mx)


def combine_files(run, dirpath,save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    newfname = os.path.join(save_dir, 'run%d.tbl'%run)

    if os.path.exists(newfname):
        return None

#   load files and sort numerically
    fnames = [ os.path.join(dirpath, f) for f in os.listdir(dirpath)
        if f.startswith('run%d_'%run) and f.endswith('h5') ]

    if not fnames:
        print("No filenames for run %d"%run)
        return None
    
    fnames = sorted(fnames, 
        key=lambda x: int(x.split('run')[-1].split('_')[-1].split('.h5')[0] ))
    

    print fnames
    data_list = []
    with h5py.File(newfname,'w') as f_out:
        for ii,ff in enumerate(fnames):
            f = h5py.File(ff,'r')

            data_list.append(f['poly_coefs'].value)

            if ii ==0:
                f_out.create_dataset('ave_bin_int', 
                    data=f['ave_bin_int'].value)
                f_out.create_dataset('num_shots_per_bin', 
                    data=f['num_shots_per_bin'].value)
                f_out.create_dataset('ref_shot_ind', 
                    data=f['ref_shot_ind'].value)
            f.close()
            

        data = np.concatenate(data_list)
        f_out.create_dataset('poly_coefs', data=data)

        del data
        
# combines files

for r in runs:
    combine_files(r, args.data_dir, args.save_dir)


