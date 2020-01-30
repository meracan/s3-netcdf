import numpy as np

def parseDescritor(idx,shape,i=0,canTuple=True):
  """
  Parsing descriptor with specific shape
  
  Parameters
  ----------
  idx: slice,int,list,ndarray or tuple.
  shape: ndarray (1D)
  i:int,optional.
    Default=0
    Axis position,starts on the left
  canTuple:bool, optional.
    Default is True.
    Can only tuple once. (0,0)=Good;(0,(0))=Bad
    
  Returns
  -------
  out : ndarray or list
  
  Notes
  -----
  First axis gets priority.
  
  Examples
  --------
  TODO
  """
  
  if isinstance(idx, slice):
    start = 0 if idx.start is None else idx.start
    stop = shape[i] if idx.stop is None else idx.stop
    step = 1 if idx.step is None else idx.step
    if (start < 0 or stop > shape[i]): raise ValueError("Exceeds limit")
    return np.arange(start, stop,step, dtype="i4")
  elif isinstance(idx, int):
    if (idx < 0 or idx >= shape[i]): raise ValueError("Exceeds limit")
    return np.array([idx],dtype="i4")
  elif isinstance(idx, list) or isinstance(idx, np.ndarray):
    return np.array(idx)
  elif isinstance(idx, tuple):
    if not (canTuple):raise TypeError("Invalid argument type.")
    array=[]
    for j, t in enumerate(idx):
      array.append(parseDescritor(t,shape,j,canTuple=False))
    return array
  else:
    raise TypeError("Invalid argument type.")  

def getIndices(idx,shape):
  """
  Parsing descriptors and creating a index table
  
  Parameters
  ----------
  idx: slice,int,list,ndarray or tuple.
  shape: ndarray (1D)
  
  Returns
  -------
  out : ndarray
  
  Notes
  -----
  First axis gets priority.
  
  Examples
  --------
  TODO
  """
  
  array = parseDescritor(idx,shape,0,canTuple=True)
  if not isinstance(array, list):array=[array]
  for i in range(len(array), len(shape)):
    # Need to fill the remainder of the array if not provided
    array.append(np.arange(0, shape[i], dtype="i4"))
  return np.array(array) 

def getMasterIndices(indices,shape,masterShape):
  """
  Get indices in master format from data
  
  Parameters
  ----------
  indices:  ndarray
  shape: ndarray (1D)
    Original data shape
  masterShape: ndarray (1D)
    Master shape
  
  Returns
  -------
  out : ndarray
  
  Notes
  -----
  TODO
  
  Examples
  --------
  TODO
  """  
  meshgrid = np.meshgrid(*indices,indexing="ij")
  index = np.ravel_multi_index(meshgrid, shape)
  index = np.concatenate(index)
  return np.array(np.unravel_index(index, masterShape)).T

def getPartitions(indices,shape,masterShape):
  """
  Find partitions based on indices
  
  Parameters
  ----------
  indices:  ndarray
  shape: ndarray (1D)
    Original data shape
  masterShape: ndarray (1D)
    Master shape
  
  Returns
  -------
  out : ndarray
  
  Notes
  -----
  TODO
  
  Examples
  --------
  TODO
  """    
  limits=[]
  n = len(shape)
  for i,step in enumerate(masterShape[n:]):
    limits.append(np.arange(np.min(indices[i]),np.max(indices[i]),step))
  limits=np.array(limits)

  masterLimits = getMasterIndices(limits,shape,masterShape)
  allPartitions=masterLimits[:, :n]
  uniquePartitions = np.unique(allPartitions,axis=0)
  
  return uniquePartitions

def dataWrapper(idx, shape,masterShape,f):
  """
  Data wrapper
  
  Parameters
  ----------
  idx:  ndarray
  shape: ndarray (1D)
    Original data shape
  masterShape: ndarray (1D)
    Master shape
  f: callback function
  Returns
  -------
  out : ndarray
  
  Notes
  -----
  TODO
  
  Examples
  --------
  TODO
  """    
  
  n = len(shape)
  indices = getIndices(idx,shape)
  partitions = getPartitions(indices, shape,masterShape)
  
  masterIndices = getMasterIndices(indices,shape,masterShape)
  
  for part in partitions:
    idata=np.all(masterIndices[:,:n] == part[None,:], axis=1)
    ipart = masterIndices[np.where(idata)[0]][:,n:]
    f(part,idata,ipart)