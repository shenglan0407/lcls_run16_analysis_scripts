
# take a run, do PCA at each q

# save the shots into train and test, save the 
import h5py
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import argparse
import numpy as np

import sys

from loki.RingData.DiffCorr import DiffCorr


parser = argparse.ArgumentParser(description='1) Compute eigenimage (principal components) by run from polar intensities\n\
2) Remove {0, 1,... num_pca} eigenimages from polar intensities up to a max number of eigenimages\n\
3) Compute difference correlation after removing eigenimages.'
,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-r','--run', type=int,
                   help='run number')
########### update this during the experiment as the types of samples we measure change
parser.add_argument('-t','--samp_type', type=int,
                   help='type of data\n \
# Sample IDs\n \
# -1: Nanoparticles small angle\n\
# -2: Nanoparticles wider angle\n\
# 0: Ga.GDP buffer\n\
# 1: Ga.Ric8 buffer\n\
# 2: Ga.GDP protein\n\
# 3: Ga.Ric8 protein\n\
# 4: Ga.GDP+AlF t= few ms\n\
# 5: Ga.GDP+AlF t= 100 ms\n\
# 6: Ga.Ric8+GTP t= few ms\n\
# 7: Ga.Ric8+GTP t= 100 ms\n\
# 8: Water \n\
# 9: Helium')
parser.add_argument('-q','--qmin', type=int,
                   help='index of minimum q to denoise and compute correlations')

parser.add_argument('-u','--qmax', type=int, default=None,
                   help='index of max q to denoise and compute correlations, or None')

parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-d','--data_dir', type=str, required=True,
                   help='where to look for the polar data')

parser.add_argument('-p','--num_pca', type=int, default=None,
                   help='num_pca+1 is the max number of eigenimages/pca components to subtract')



########### update this during the experiment as the types of samples we measure change
def sample_type(x):
    return {-1:'NP_sml',
    -2:'NP_wid',
     0:'GaGDP_buf',
     1:'GaRic8_buf',
     2:'GaGDP_pro',
     3:'GaRic8_pro',
     4:'GaGDP_AlF_short',
     5:'GaGDP_AlF_long',
     6:'GaRic8_GTP_short',
     7:'GaRic8_GTP_long',
     8:'h2o',
     9:'he'}[x]
###############change this if the total number of sample types measured changes
samp_type_range= range(-1,10)


def normalize_shot(ss, this_mask):
    if ss.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        ss = ss.astype(np.float64)
    
    ss *=this_mask
    mean_ss = ss.sum(-1)/this_mask.sum(-1) 
    ss = ss-mean_ss[:,None]
    return np.nan_to_num(ss*this_mask)

def reshape_unmasked_values_to_shots(shots,mask):
    # this takes vectors of unmasked values, and reshaped this into their masked forms
    # mask is 2D, shots are 1D
    assert(shots.shape[-1]==np.sum(mask) )
    flat_mask = mask.flatten()
    reshaped_shots = np.zeros( (shots.shape[0],mask.size), dtype=shots.dtype)
    
    reshaped_shots[:, flat_mask==1] = shots
    
    return reshaped_shots.reshape((shots.shape[0],mask.shape[0],mask.shape[1]))
    


args = parser.parse_args()


run_num = args.run

if args.samp_type not in samp_type_range:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file

data_dir = args.data_dir
save_dir = args.out_dir

if args.num_pca is None:
    num_pca_file = os.path.join(save_dir,'num_pca_components.txt')

    if not os.path.exists(num_pca_file):
        print("there is no num_pca_components.txt file in %s"%save_dir)
        sys.exit()
    num_pca_components = np.loadtxt(num_pca_file)


save_dir = os.path.join( args.out_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print save_dir


run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')

# output file to save data
out_file = run_file.replace('.tbl','_PCA-denoise.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'a')

if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    #every run should have it's own binned polar mask saved, if not, something is run, exit
    print("this run has no binned mask. Something is wrong. Quit!")
    print(os.path.join(data_dir, run_file))
    sys.exit()

PI = f['polar_imgs']
shot_tags = np.arange(0,PI.shape[0])

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
    print('PCA denoising for qidx %d'%qidx)
    q_group = 'q%d'%qidx
    if q_group not in f_out.keys():
        f_out.create_group(q_group)
    shots=PI[:,qidx,:][:,None,:]
    this_mask = mask[qidx][None,:]
    print('normaling shots...')
    norm_shots = np.zeros_like(shots)
    for idx,ss in enumerate(shots):
        norm_shots[idx]=normalize_shot(ss,this_mask)
    # do we want to normalize by the entire range of intensity?
    # divide into Train and test
    num_shots = norm_shots.shape[0]
    cutoff = int(num_shots*0.1) # use 10% of the shots as testing set
    partial_mask = this_mask.copy()
    Train = norm_shots[cutoff:, partial_mask==1]
    Test = norm_shots[:cutoff, partial_mask==1]

    print ("%d test shots"%(Test.shape[0]))
    print ("%d train shots"%(Train.shape[0]))
   

    qvalues = np.linspace(0,1,partial_mask.shape[0])
    
    if args.num_pca is None:
        num_pca = int(num_pca_components[qidx])
        max_pca = num_pca+5
    else:
        num_pca = args.num_pca+1
        max_pca = args.num_pca+1

    print('denoisng with PCA critical num_pca_components = %d...'%num_pca)
    if 'pca_components' not in f_out[q_group].keys():
        # if there is no pca component saved, then run it and save the components
        pca=PCA(n_components=50, whiten = False)

        new_Train=pca.fit_transform(Train)
        new_Test = pca.transform(Test)
        if 'explained_variance_ratio' not in f_out[q_group].keys():
            f_out.create_dataset('q%d/explained_variance_ratio'%qidx,data=pca.explained_variance_ratio_)

        # get back the masked images and components
        components=pca.components_
        f_out.create_dataset('q%d/pca_components'%qidx,data=components)
    else:
        #load the components previously save the then do the transformations
        components = f_out['q%d/pca_components'%qidx].value
        _m = Train.astype(np.float64).mean(0)
        new_Train = (Train.astype(np.float64)-_m).dot(components.T)

        _m = Test.astype(np.float64).mean(0)
        new_Test = (Test.astype(np.float64)-_m).dot(components.T)
    # get back the masked images and components
 
    masked_mean_train =reshape_unmasked_values_to_shots(Train,partial_mask).mean(0)
    masked_mean_test =reshape_unmasked_values_to_shots(Test,partial_mask).mean(0)

    # denoise
    for nn in range(max_pca):
        pca_group = 'q%d/pca%d'%(qidx,nn)
        if 'pca%d'%nn not in f_out[q_group].keys():
            f_out.create_group(pca_group)
        else:
            print("pca denoise at pca n_components = %d is already done. Skip!"%nn)
            continue

        if nn>0:

            print('subtracting noise with pca n_components =  %d'%nn)
            Test_noise = new_Test[:,:nn].dot(components[:nn])
            denoise_Test = reshape_unmasked_values_to_shots(Test-Test_noise-Test.mean(0)[None,:],
            partial_mask)
            Train_noise = new_Train[:,:nn].dot(components[:nn])
            denoise_Train= reshape_unmasked_values_to_shots(Train-Train_noise-Train.mean(0)[None,:]
                                                        , partial_mask)

            
            dc=DiffCorr(denoise_Train,qvalues,0,pre_dif=False)
            Train_difcor= (dc.autocorr()).mean(0)

            dc=DiffCorr(denoise_Test,qvalues,0,pre_dif=False)
            Test_difcor= (dc.autocorr()).mean(0)


            f_out.create_dataset('q%d/pca%d/test_difcor'%(qidx,nn)
                ,data=Test_difcor)
            f_out.create_dataset('q%d/pca%d/train_difcor'%(qidx,nn)
                ,data=Train_difcor)
    
        else:
            print('not doing denoising, just computing baseline')

            dc=DiffCorr(norm_shots[cutoff:],qvalues,0,pre_dif=False)
            difcor= (dc.autocorr()).mean(0)
            f_out.create_dataset('q%d/pca%d/train_difcor'%(qidx,nn)
            ,data=difcor)
            
            dc=DiffCorr(norm_shots[:cutoff],qvalues,0,pre_dif=False)
            difcor= (dc.autocorr()).mean(0)
            f_out.create_dataset('q%d/pca%d/test_difcor'%(qidx,nn)
                ,data=difcor)
    
    if 'num_shots' not in f_out[q_group].keys():        
        f_out.create_dataset('q%d/num_shots'%qidx, data=norm_shots.shape[0])
    del shots
    del norm_shots

print ("done!")
f_out.close()