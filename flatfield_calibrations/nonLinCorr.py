"""
Python example to correct for image spatial dependent non-linearity. 
Use function "getCorrectionFunc" in order to generate a correction 
function from a intensity dependence measurement of an identical 
signal.
"""

import numpy as np
import copy
from scipy import linalg

def iterfy(iterable):
    if isinstance(iterable, basestring):
        iterable = [iterable]
    try:
        iter(iterable)
    except TypeError:
        iterable = [iterable]
    return iterable

def polyVal(comps,i0):
  """Multidimensional version of numpy.polyval"""                                                          
  i0 = np.asarray(iterfy(i0))                                             
  pol = np.vander(i0,len(comps))                                        
  return np.asarray(np.matrix(pol)*np.matrix(comps.reshape((len(comps),-1)))).reshape((len(i0),)+np.shape(comps)[1:])                            
                                                                                
def polyDer(comps,m=1):                                                         
  """Multidimensional version of numpy.polyder"""                                                          
  compsf = comps.reshape((len(comps),-1))
  n = len(compsf) - 1                                                            
  y = compsf.reshape((n+1,-1))[:-1] * np.expand_dims(np.arange(n, 0, -1),1)                      
  if m == 0:                                                                    
    val = comps
    return val
  else:                                                                         
    val = polyDer(y, m - 1) 
    return val.reshape((n,)+np.shape(comps)[1:])

def polyFit(i0,Imat,order=3, removeOrders=[]):
  """Multidimensional version of numpy.polyfit"""                                                          
  Imatf = Imat.reshape((len(Imat),-1))
  pol = np.vander(i0,order+1)                                                   
  removeOrders = iterfy(removeOrders)                                     
  removeOrders = np.sort(removeOrders)[-1::-1]                                  
  for remo in removeOrders:                                                     
    pol = np.delete(pol,-(remo+1),axis=1)                                       
  lhs = copy.copy(pol)                                                          
  scale = np.sqrt((lhs*lhs).sum(axis=0))                                        
  lhs /= scale                                                                  
  comps,resid,rnk,singv = linalg.lstsq(lhs,Imatf)                                
  comps = (comps.T/scale).T                                                     
                                                                                
  for remo in removeOrders:                                                     
    comps = np.insert(comps,order-remo,0,axis=0)                                
  return comps.reshape((order+1,)+np.shape(Imat)[1:])

def getCorrectionFunc(dmat=None,i=None,order=5,search_dc_limits=None):
  """ 
  Create nonlinear correction function from a calibration dataset consiting of:
    i     	array of intensity values (floats) of the calibration
    dmat   	ND array of the reference patterns corresponding to values of i,
                The first dimension corresponds to the calibration intensity 
		values and has the same length as i.
    
    order	the polynomial order up to which the correction will be 
                deveoped.

    search_dc_limits	absolute limits around ic which are used to determine the 
    		calibration value of ic as linear approximation of a short interval. 
		optional, can sometimes help to avoid strong deviations of the 
		polynomial approximatiuon from the real measured points.
  
  Returns corrFunc(D,i), a function that takes an ND array input for correction
                (1st dimension corresponds to the different intensity values) 
		as well as the intensity array i.
  """
  if search_dc_limits is not None:
    search_dc_limits = iterfy(search_dc_limits)
    if len(search_dc_limits)==1:
      msk = (i>i-np.abs(search_dc_limits)) & (i<i+np.abs(search_dc_limits))
    elif len(search_dc_limits)==2:
      msk = (i>i-np.min(search_dc_limits)) & (i<i+np.max(search_dc_limits))
    p0 = tools.polyFit(i[msk],dmat[msk,...],2)
    dc = tools.polyVal(p0,i0_wp)
    
    pc = tools.polyFit(i,Imat-dc,order,removeOrders=[0])
    c = lambda(i): polyVal(pc,i) + dc
  else:
    pc = polyFit(i,dmat,order,removeOrders=[])
    c = lambda(i): polyVal(pc,i)

  return c

def getCorrectionFunc2(dmat=None,i=None,order=5,search_dc_limits=None):
  """ 
  Create nonlinear correction function from a calibration dataset consiting of:
    i     	array of intensity values (floats) of the calibration
    dmat   	ND array of the reference patterns corresponding to values of i,
                The first dimension corresponds to the calibration intensity 
		values and has the same length as i.
    
    order	the polynomial order up to which the correction will be 
                deveoped.

    search_dc_limits	absolute limits around ic which are used to determine the 
    		calibration value of ic as linear approximation of a short interval. 
		optional, can sometimes help to avoid strong deviations of the 
		polynomial approximatiuon from the real measured points.
  
  Returns corrFunc(D,i), a function that takes an ND array input for correction
                (1st dimension corresponds to the different intensity values) 
		as well as the intensity array i.
  """
  if search_dc_limits is not None:
    search_dc_limits = iterfy(search_dc_limits)
    if len(search_dc_limits)==1:
      msk = (i>i-np.abs(search_dc_limits)) & (i<i+np.abs(search_dc_limits))
    elif len(search_dc_limits)==2:
      msk = (i>i-np.min(search_dc_limits)) & (i<i+np.max(search_dc_limits))
    p0 = tools.polyFit(i[msk],dmat[msk,...],2)
    dc = tools.polyVal(p0,i0_wp)
    
    pc = tools.polyFit(i,Imat-dc,order,removeOrders=[0])

  else:
    pc = polyFit(i,dmat,order,removeOrders=[])


  return pc

