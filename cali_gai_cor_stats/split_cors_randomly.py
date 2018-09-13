import numpy as np
import h5py
import os

import argparse
import sys
import glob

from loki.RingData.DiffCorr import DiffCorr
import glob

parser = argparse.ArgumentParser(description='randomly split into two dif cors')
parser.add_argument('-s', '--sample', type=str, required=True, help='run number to process')
parser.add_argument('-d', '--data_dir', type=str,
    default = '/reg/d/psdm/cxi/cxilr6716/scratch/pca_denoise/cali_gai_filtered2/',
help='dir in which the pca denoised data are')
parser.add_argument('-o', '--out_dir', type=str,required=True,
help='dir in which to store the output')
parser.add_argument('-m', '--mask_dir', type=str,
    default='/reg/d/psdm/cxi/cxilr6716/scratch/combined_tables/cali_gai/',
help='dir in which the binned masks are, i.e. the polar image directory')


args = parser.parse_args()


sample = args.sample
data_dir = args.data_dir
save_dir =args.out_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print "saving in %s..."%save_dir

all_files =glob.glob('%s/*PCA*'%os.path.join(data_dir,sample))
print all_files
run_nums = [int(os.path.basename(ff).split('_')[0].split('n')[-1]) for ff in all_files]
run_nums = sorted(list(set(run_nums)))

outfname = os.path.join(save_dir,'%s_random_split.h5'%sample)

# if os.path.isfile(outfname):
#     print ('will not overwrite file')
#     sys.exit()

with h5py.File(outfname,'a') as f_out:
    for run in run_nums:
        print("splitting shots for run %d "%(run) )
        flist = glob.glob('%s/run%d_PCA-denoise*.h5'%(os.path.join(data_dir,sample),run))
        n_files = len(flist)
        
        for file_num in range(n_files):
            f = h5py.File('%s/run%d_PCA-denoise_%d.h5'%(os.path.join(data_dir,sample),
                run,file_num),'r')
            try:
                pca_num = f['q10']['num_pca_cutoff'].value
            except KeyError:
                print("skipping run %d"%run)
                continue

            if 'run%d_%d'%(run,file_num) in f_out.keys():
                print("already seen this run, skip!")
                continue

            ##### load the mask used for this run
            f_mask = h5py.File(os.path.join(args.mask_dir,'run%d.tbl'%run),'r')

            mask = f_mask['polar_mask_binned'].value
            mask = (mask==mask.max())
            mask.shape
            # do the mask cor
            qs = np.linspace(0,1,mask.shape[0])
            dc=DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
            mask_cor = dc.autocorr().mean(0)
            
            f_mask.close()

            f_out.create_group('run%d_%d'%(run,file_num))

            all_ave_cors=[]
            all_err=[]

            all_nums=[]



            n_q = len([kk for kk in f.keys() if kk.startswith('q')])
            for qidx in range(n_q):
                print('run%d file %d q%d'%(run,file_num,qidx))
                if 'num_pca_cutoff2' in f['q%d'%qidx].keys():
                    pca_num = f['q%d'%qidx]['num_pca_cutoff2'].value
                else:
                    pca_num = f['q%d'%qidx]['num_pca_cutoff'].value
                #take half of everything
                num = f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'].shape[0]

                chunk_cor = []
                chunk_err=[]
                chunk_num = []

                a = range(f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'].shape[0])
                np.random.shuffle(a)

                f_out.create_dataset('run%d_%d/q%d_train_random_inds'%(run,file_num,qidx)
                    ,data=a)
                x1=sorted(a[:len(a)/2])
                x2=sorted(a[len(a)/2:])
                x_list =[x1,x2]

                for x_inds in x_list:
                    cc = f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'].value[x_inds,0]/mask_cor[qidx][None,:]
                    ave = cc.mean(0)
                    err = cc.std(0)/np.sqrt(float(cc.shape[0]))
                    chunk_cor.append(ave)
                    chunk_err.append(err)
                    chunk_num.append(len(x_inds))

                all_ave_cors.append(np.array(chunk_cor))
                all_err.append(np.array(chunk_err))
                all_nums.append(np.array(chunk_num))


            f_out.create_dataset('run%d_%d/ave_cor_byChunk'%(run,file_num),
                data=np.array(all_ave_cors))
            f_out.create_dataset('run%d_%d/err_byChunk'%(run,file_num),
                data=np.array(all_err))
            f_out.create_dataset('run%d_%d/num_shots'%(run,file_num),
                data=np.array(all_nums))

            

            # print np.array(all_ave_cors).shape
    ############################ shouldn't need to change this much
    ##### chunk the errors as well
    # aggregate all the averages
    print('aggregating results')
    ave_cor1 =[]
    ave_err=[]
    total_shots1=[]
    num_chunks=None
    keys=[kk for kk in f_out.keys() if kk.startswith('run')]

    for kk in keys:
        if num_chunks is None:
            num_chunks = f_out[kk]['ave_cor_byChunk'].value.shape[1]
        ave_cor1.append(f_out[kk]['ave_cor_byChunk'].value)
        ave_err.append(f_out[kk]['err_byChunk'].value)
        total_shots1.append(f_out[kk]['num_shots'].value)
    
    ave_cor1=np.array(ave_cor1)
    ave_err = np.array(ave_err)
    total_shots1=np.array(total_shots1)
    
    chunk_ave_cors =[]
    chunk_err =[]
    chunk_nShots=[]
    for ic in range(num_chunks):
        shots = ave_cor1[:,:,ic,:]
        errs = ave_err[:,:,ic,:]
        nn = total_shots1[:,:,ic]

        cor1 = (shots *(nn.astype(float)/nn.sum(0)[None,:])[:,:,None]).sum(0)
        ee = np.sqrt((errs**2 *( nn**2./(nn.sum(0)[None,:])**2.)[:,:,None]).sum(0))
        
        chunk_ave_cors.append(cor1)
        chunk_err.append(ee)
        chunk_nShots.append(nn)

    chunk_ave_cors = np.array(chunk_ave_cors)
    chunk_err = np.array(chunk_err)
    chunk_nShots = np.array(chunk_nShots)

    f_out.create_dataset('running_ave_cor',data=chunk_ave_cors)
    f_out.create_dataset('running_ave_err',data=chunk_err)
    f_out.create_dataset('running_num_shots',data=chunk_nShots)

    print('Done!')