import numpy as np

def flatten_and_mask_shots(shot, mask):
    
    size = shot.shape[0]*shot.shape[1]
    flat_mask = mask.reshape(size)
    flat_shot =  shot.reshape(size)*flat_mask
    
    return flat_shot[flat_mask]

def unflatten_shots(flat_shots,mask):
    num_shots = flat_shots.shape[0]
    shape = mask.shape
    flat_mask = mask.reshape(shape[0]*shape[1]
                            )
    shots = np.zeros( (num_shots,shape[0]*shape[1]), dtype = flat_shots.dtype)
    shots[:, flat_mask ] = flat_shots
    
    return shots.reshape( (num_shots,shape[0],shape[1]))

def norm_w_mask(x,mask, threshold):
    x_out = np.zeros_like(x, dtype=np.float64)
    mask= mask.astype(bool)
    for ii in range(x.shape[0]):
        # if np.nan_to_num(x[ii])
        k=np.median(x[ii][mask[ii]])
        mu = np.std(x[ii][mask[ii]])
        this_mask = (np.abs(x[ii] - k)<mu*threshold)*mask[ii]

        shot_mean = x[ii][this_mask].mean()
        # y = (x[ii]/shot_mean)*this_mask
        # shot_mean = y[this_mask].mean()
        # x_out[ii] = (y-shot_mean)* this_mask
        x_out[ii] = (x[ii]-shot_mean)* this_mask
        
    return x_out

def diff_all_shots(images,mask,threshold):
    images = norm_all_shots(images, mask, threshold)
    diffs = images - images.mean(0)[None,:]
    
    return diffs, images

def norm_all_shots(images,mask, threshold):
    norms = np.zeros_like(images, dtype=np.float64)
    for ii in range(norms.shape[0]):
        norms[ii]=norm_w_mask(images[ii],mask, threshold)
    return norms