import numpy as np

def getChildShape(dataShape,dtype="f8",maxSize=1):
  '''
  TODO: Change description
  Find new array shape based on maximum file size
  Last dimensions gets priority
  :param data: numpy.array
  :param maxSize: float (MB)
  :return: numpy.array (shape)
  '''
  
  itemSize = np.dtype(dtype).itemsize
  maxSize = maxSize * 1024.0**2
  
  items = 1
  fileShape = np.ones(len(dataShape), dtype=np.int)
  for i in range(len(dataShape) - 1, -1, -1):
    n = dataShape[i]
    items *= n
    fileSize = items * itemSize
    p = np.int(np.ceil(fileSize / maxSize))
    if (p > 1):
      fileShape[i] = np.int(np.ceil(n * 1.0 / p))
      break
    else:
      fileShape[i] = n
  
  return fileShape


def getMasterShape(dataShape,return_childShape=False,**kwargs):
  """
  Find array shape based on new file partitions
  
  Parameters
  ----------
  dataShape: ndarray
  
  Returns
  -------
  shape : ndarray
  
  Notes
  -----
  
  Examples
  --------
  i.e (10,10)=> (10,1,1,10); 10 partitions on the first axis with file shape of (1,10)
  i.e (10,10)=> (10,2,1,5); 10 partitions on the first axis, 2 partions on the second axis with file shape of (1,5)
  
  """

  fileShape = getChildShape(dataShape,**kwargs)
  partitions = np.ceil(np.array(dataShape) / fileShape).astype('int')
  masterShape = np.insert(fileShape, 0, partitions)
  if(return_childShape):return masterShape, fileShape
  return masterShape

# def getMasterIndices(indices,dataShape,masterShape):
#   """
#   TODO: Change description
#   :param indices:
#   :param dataShape:
#   :param masterShape:
#   :return:
#   """
  
#   indices = np.array(indices)
#   if(len(indices.shape)==1):indices=indices[np.newaxis,:]
#   indices = indices.T
#   index = np.ravel_multi_index(indices, dataShape)
#   return np.array(np.unravel_index(index, masterShape)).T

def getMasterIndices(idx,shape,masterShape):
  indices = createIndices(shape,idx)
  
  meshgrid = np.meshgrid(*indices,indexing="ij")
  index = np.ravel_multi_index(meshgrid, shape)
  index = np.concatenate(index)
  return np.array(np.unravel_index(index, masterShape)).T

def checkSize(size,maxSize=1E+9):
    if(size>1E+9):raise ValueError("Array too large")

def __getIndices(shape,idx,i=0,canTuple=True):
  if isinstance(idx, slice):
    start = 0 if idx.start is None else idx.start
    end = shape[i] if idx.stop is None else idx.stop
    step = 1 if idx.step is None else idx.step
    if (end > shape[i]): raise ValueError("Exceeds limit")
    return np.arange(start, end,step, dtype="i4")
  elif isinstance(idx, int):
    if (idx >= shape[i]): raise ValueError("Exceeds limit")
    return np.array([idx],dtype="i4")
  elif isinstance(idx, list) or isinstance(idx, np.ndarray):
    return np.array(idx)
  elif isinstance(idx, tuple):
    if not (canTuple):raise TypeError("Invalid argument type.")
    array=[]
    for j, t in enumerate(idx):
      array.append(__getIndices(shape,t,j,canTuple=False))
    return array
  else:
    raise TypeError("Invalid argument type.")  

def createIndices(shape,idx):
  array = __getIndices(shape,idx,0,canTuple=True)
  if not isinstance(array, list):array=[array]
  for j in range(len(array), len(shape)):
    array.append(np.arange(0, shape[j], dtype="i4"))
  array = np.array(array) 
  checkSize(np.prod(array.shape))
  return array

# def getPartitions(indices, dataShape,masterShape):
#   """
#   TODO: Change description
#   :param indices:
#   :param dataShape:
#   :param masterShape:
#   :return:
#   """
#   masterIndices = getMasterIndices(indices, dataShape,masterShape)
#   n = len(dataShape)
#   allPartitions=masterIndices[:, :n]
#   uniquePartitions,indexPartitions = np.unique(allPartitions,axis=0,return_inverse=True)
#   indexDataFile = masterIndices[:, n:]
#   indexData = np.concatenate((indexPartitions[:, None], indexDataFile), axis=1)
#   return uniquePartitions,indexPartitions,indexData

def getPartitions(idx, shape,masterShape):
  """
  TODO: Change description
  :param indices:
  :param dataShape:
  :param masterShape:
  :return:
  """
  
  masterIndices = getMasterIndices(idx, shape,masterShape)
  n = len(shape)
  allPartitions=masterIndices[:, :n]
  uniquePartitions,indexPartitions = np.unique(allPartitions,axis=0,return_inverse=True)
  indexDataFile = masterIndices[:, n:]
  indexData = np.concatenate((indexPartitions[:, None], indexDataFile), axis=1)
  
  return uniquePartitions,indexPartitions,indexDataFile

# def getMasterSlices(idx,dataShape,masterShape):
#   """
#   TODO: Change description
#   :param indices:
#   :param dataShape:
#   :param masterShape:
#   :return:
#   """
#   if isinstance(idx, slice):idx=[idx]
#   if isinstance(idx, int):idx=[idx]
#   indices=[]
#
#   for dim in range(len(dataShape)):
#     start=0
#     stop=dataShape[dim]-1
#     if dim<len(idx):
#       value=idx[dim]
#       if isinstance(value, list): raise Exception("Don't not support fancy indexing at the moment")
#       if isinstance(value, int):value=slice(value,value,None)
#
#       if value.start is not None:start=value.start
#       if value.stop is not None:stop=value.stop
#     indices.append([start,stop])
#
#
#
#   indices = np.array(indices)
#
#   indices = indices
#   index = np.ravel_multi_index(indices, dataShape)
#   return np.array(np.unravel_index(index, masterShape)).T

# def getPartitionsSlices(idx,dataShape,masterShape):
#   masterIndices = getMasterSlices(idx, dataShape,masterShape)
#   n = len(dataShape)
#   partitions=masterIndices[:, :n]
#   spart=partitions[0]
#   epart = partitions[1]
#   indexDataFile = masterIndices[:, n:]
#   sdpart = indexDataFile[0]
#   edpart = indexDataFile[1]
#
#   index = np.ravel_multi_index(np.array([spart, epart]).T, masterShape[:n])
#
#   dataIndex = masterShape[n:]
#   startIndex = dataIndex * 0
#   indexfiles = np.arange(index[0], index[1] + 1)
#
#   partitionRange = []
#   for i in indexfiles:
#     fileIndex = np.array(np.unravel_index(i, masterShape[:n])).T
#     if (i == index[0]):
#       # Start
#       partitionRange.append(dict(partition=fileIndex,start=sdpart,end=dataIndex))
#     elif (i == index[1] ):
#       # End
#       partitionRange.append(dict(partition=fileIndex,start=startIndex, end=edpart))
#     else:
#       partitionRange.append(dict(partition=fileIndex,start=startIndex, end=dataIndex))
#
#
#
#
#   return partitionRange


def concatenatePartitions(partitions):
  partitions =  np.array(partitions)
  return partitions

# def detData():

def indexMulti(data,indices):
  """
  TODO: Change description
  :param data:
  :param indices:
  :return:
  """
  data=np.array(data)
  indices = np.array(indices)
  
  n= len(data.shape)
  if(len(indices.shape)!=2):
    raise ValueError("Indices shape needs to be 2D")
  if len(data.shape) != indices.shape[1]:
    raise ValueError("Shapes are not identical")
  
  if (n == 1):return data[indices[:, 0]]
  if (n == 2): return data[indices[:, 0],indices[:, 1]]
  if (n == 3): return data[indices[:, 0], indices[:, 1], indices[:, 2]]
  if (n == 4): return data[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
  if (n == 5): return data[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], indices[:, 4]]
  raise ValueError("Function not design for more than 5 dimensions")