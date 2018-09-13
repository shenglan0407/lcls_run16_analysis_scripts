import h5py
import numpy as np

from sklearn.decomposition import NMF

from scipy.interpolate import interp1d

from loki.RingData import DiffCorr
import argparse
import os

def reorder_data(data, exp_cpsi):
    n_shots=data.shape[0]
    order=np.argsort(exp_cpsi)
    exp_cpsi = np.array(sorted(exp_cpsi))
    for ii in range(n_shots):
        data[ii] = data[ii][order]
    return data, exp_cpsi

def interp_shots(norm_X2, interp_num_phi,old_cpsi,new_cpsi):
    
    interp_X = np.zeros( (norm_X2.shape[0],interp_num_phi) )

    for ii in range(interp_X.shape[0]):
        f = interp1d(old_cpsi, norm_X2[ii])
        try:
            interp_X[ii] = f(new_cpsi)
        except:
            print ii
            break
    return interp_X

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

parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-m','--mask_dir', type=str, required=True,
                   help='where to look for the polar data')

parser.add_argument('-d','--data_dir', type=str, required=True,
                   help='where to look for the single-shot diff cor')

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
run_num = args.run

if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file

data_dir = os.path.join(args.data_dir, sample)
save_dir = os.path.join( args.out_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print save_dir

# load and compute mask dif cor
f_mask = h5py.File(os.path.join(args.mask_dir,'run%d.tbl'%run_num),'r')

mask = f_mask['polar_mask_binned'].value
mask = (mask==mask.max())
mask.shape
qs = np.linspace(0,1,mask.shape[0])
dc=DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
mask_cor = dc.autocorr().mean(0)
f_mask.close()

##parameters
phi_offset=15
interp_num_phi =100
n_comp = 20

# load simulations and interpolate simulations and data
if sample.startswith('GDP'):
    sims = np.load('/reg/d/psdm/cxi/cxilr6716/results/nnmf_filter/GDP_closed_121models.npy')
else:
    sims = np.load('/reg/d/psdm/cxi/cxilr6716/results/nnmf_filter/GDP-AlF_closed_140models.npy')

all_exp_cpsi = np.load('/reg/d/psdm/cxi/cxilr6716/results/nnmf_filter/exp_cpsi.npy')
all_sim_cpsi = np.load('/reg/d/psdm/cxi/cxilr6716/results/nnmf_filter/sim_cpsi.npy')

for file_num in range(args.n_files):

    with h5py.File( os.path.join (save_dir, 'run%d_nnmf_filtered_%d.h5'%(run_num,file_num) ), 'a') as f_out: 

        f = h5py.File(os.path.join(data_dir,'run%d_PCA-denoise_%d.h5'%(run_num,file_num)), 'r')
        qinds = sorted([int(kk.split('q')[-1]) for kk in f.keys() if kk.startswith('q')])

         # noised filtered average shot, and errors
        pro=np.zeros( (len(qinds), interp_num_phi), dtype=np.float64)
        err = np.zeros( (len(qinds), interp_num_phi), dtype=np.float64)
        
        num_phi = f['q0']['pca0']['train_difcor'].shape[-1]
        original_pro=np.zeros( (len(qinds), num_phi/2- phi_offset*2), dtype=np.float64)
        original_err = np.zeros( (len(qinds), num_phi/2 - phi_offset*2), dtype=np.float64)
        interp_cpsi = np.zeros( (len(qinds), interp_num_phi), dtype=np.float64)

        for qidx in qinds:
            if 'q%d'%qidx in f_out.keys():
                print('already filtered for q%d. Skip!'%qidx)
                continue
            print('filtering for q%d...'%qidx)
            grp = f_out.create_group('q%d'%qidx)

            cutoff=f['q%d'%qidx]['num_pca_cutoff'].value
            shots=(f['q%d'%qidx]['pca%d'%cutoff]['all_train_difcors'][:]/mask_cor[qidx])[:,0,:]
            num_phi = shots.shape[-1]
            
            X = sims[:,qidx,phi_offset:num_phi/2-phi_offset]
            #### make everything positive
            shots = shots[:,phi_offset:num_phi/2-phi_offset]
            shots -= shots.min(-1)[:,None]

            original_pro[qidx] = shots.mean(0)
            original_err[qidx] = shots.std(0)/np.sqrt(shots.shape[0])

            print('interpolating shots and simulations...')
            exp_cpsi = all_exp_cpsi[qidx,phi_offset:num_phi/2-phi_offset]
            sim_cpsi = all_sim_cpsi[qidx,phi_offset:num_phi/2-phi_offset]

            # reorder
            shots, exp_cpsi2 = reorder_data(shots,exp_cpsi)
            X, sim_cpsi2 = reorder_data(X,sim_cpsi)
            # interpolate
            
            new_cpsi = np.linspace(np.max( (exp_cpsi2.min(),sim_cpsi2.min()) )+0.05,
                                  np.min((exp_cpsi2.max(), sim_cpsi2.max()))-0.05,
                                  interp_num_phi,endpoint=False )
           
            interp_X = interp_shots(X, interp_num_phi, sim_cpsi2, new_cpsi)
            interp_pro = interp_shots(shots, interp_num_phi, exp_cpsi2, new_cpsi)
            
            print('constructing filter...')
            # transform and inverse transform
            model = NMF(n_components=n_comp,solver='cd')
            W=model.fit_transform(interp_X)
            H=model.components_

            new_pro = model.transform(interp_pro)
            inverse_pro = model.inverse_transform(new_pro)

            # average and error estimate

            pro[qidx] = inverse_pro.mean(0)
            err[qidx] = inverse_pro.std(0)/np.sqrt(inverse_pro.shape[0])
            interp_cpsi[qidx] = new_cpsi
            grp.create_dataset('all_filtered_cor',data=inverse_pro)
            grp.create_dataset('W_matrix',data=new_pro)
            grp.create_dataset('H_matrix',data=H)


        f_out.create_dataset('ave_cor',data=pro)
        f_out.create_dataset('err',data=err)

        f_out.create_dataset('unfiltered_ave_cor',data=original_pro)
        f_out.create_dataset('unfiltered_err',data=original_err)
        f_out.create_dataset('interp_cpsi', data = interp_cpsi)

        f_out.create_dataset('cpsi', data = all_exp_cpsi[:,phi_offset:num_phi/2-phi_offset])
        f_out.create_dataset('nnmf_n_components', data = model.n_components)

        f.close()
