
import numpy as np
from numpy import polyval
from joblib import Parallel, delayed

import time

def calibrate_chunk(delta,coefs,intensity):
    return [polyval(coefs[ii,:], intensity) for ii in range(delta)]


if __name__ == '__main__':
    num = 200000
    chunks = 10
    delta = num/chunks
    starts=np.arange(0,num,delta)


    coefs = np.random.rand(num,10)
    intensity = 5.

    tic = time.time()
    calibrated_shot1= np.array([polyval(coefs[ii,:], intensity) for ii in range(coefs.shape[0])] )
    # print calibrated_shot1.shape
    toc = time.time()
    d=toc-tic
    print ("NOT Parallel time %.2f"%d)

    tic = time.time()
    # calibrated_shot=Parallel(n_jobs=8)(delayed(polyval)(coefs[ii,:], intensity) 
    #     for ii in range(coefs.shape[0]))

    calibrated_shot2=Parallel(n_jobs=8)(delayed(calibrate_chunk)(delta,coefs[s:(s+delta)], intensity) 
        for s in starts)
    # print len(calibrated_shot)
    # calibrated_shot2=np.concatenate(calibrated_shot2)
    # print calibrated_shot2.shape
    toc = time.time()
    d=toc-tic

    print ("Parallel time %.2f"%d)

    # print np.all(calibrated_shot1==calibrated_shot2)
    # print np.allclose(calibrated_shot1,calibrated_shot2)