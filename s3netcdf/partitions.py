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

def getMasterIndices(indices,dataShape,masterShape):
  """
  TODO: Change description
  :param indices:
  :param dataShape:
  :param masterShape:
  :return:
  """
  
  indices = np.array(indices)
  if(len(indices.shape)==1):indices=indices[np.newaxis,:]
  indices = indices.T
  index = np.ravel_multi_index(indices, dataShape)
  return np.array(np.unravel_index(index, masterShape)).T

# def createIndices(shape,offsets=None):
#     transposeShape = np.append(np.arange(1,len(shape)+1),0)
#     indices = np.indices(shape).transpose(transposeShape).reshape(np.prod(shape), len(shape))
#
#     if offsets is None:return indices
#
#     if(len(shape)!=len(offsets)):raise ValueError("shape length must be equal to offsets length")
#     for i in offsets:
#       indices[:,i] = indices[:,i] + offsets[i]
#     return indices

def checkSize(size,maxSize=1E+9):
    if(size>1E+9):raise ValueError("Array too large")

def _getIndices(shape,idx,i=0,canTuple=False):
  array = []
  if isinstance(idx, slice):
    start = 0 if idx.start is None else idx.start
    end = shape[i] if idx.stop is None else idx.stop
    step = 1 if idx.step is None else idx.step
    checkSize((end - start) * np.prod(shape[slice(i + 1,None)]))
    
    array.append(np.arange(start, end,step, dtype="i4"))
    for j in range(i+1, len(shape)):
      array.append(np.arange(0, shape[j], dtype="i4"))
  elif isinstance(idx, int):
    checkSize(np.prod(shape[slice(i + 1)]))
    
    array.append(idx)
    for j in range(i+1, len(shape)):
      array.append(np.arange(0, shape[j], dtype="i4"))
  elif isinstance(idx, list) or isinstance(idx, np.ndarray):
    t = np.array(idx)
    checkSize(t.size)
    
    array.append(t)
    if (len(t) > shape[i]): raise ValueError("Length of exceeds limit")
    for j in range(i+1, len(shape)):
      array.append(np.arange(0, shape[j], dtype="i4"))
  elif isinstance(idx, tuple):
    if not (canTuple):raise TypeError("Invalid argument type.")
    for j, t in enumerate(idx):
      array.append(_getIndices(shape,t,j))
  else:
    raise TypeError("Invalid argument type.")
  return array
  
def createIndices(shape,idx):
  return _getIndices(shape,idx)
  
  # array = []
  # if isinstance(idx, slice):
  #   start = 0 if slice.start is None else slice.start
  #   end = shape[0] if slice.stop is None else slice.stop
  #   step = 1 if slice.step is None else slice.step
  #   size = (end - start) * np.prod(shape[1:])
  #   checkSize(size)
  #   array.append(np.arange(start, end, dtype="i4"))
  #   for i in range(1,len(shape)):
  #     array.append(np.arange(0, shape[i], dtype="i4"))
  #
  # elif isinstance(idx, int):
  #   size = np.prod(shape[1:])
  #   checkSize(size)
  #   array.append(idx)
  #   for i in range(1,len(shape)):
  #     array.append(np.arange(0, shape[i], dtype="i4"))
  #
  # elif isinstance(idx, tuple):
  #   for i, t in enumerate(idx):
  #     if isinstance(t, slice):
  #       start = 0 if slice.start is None else slice.start
  #       end = shape[i] if slice.stop is None else slice.stop
  #       step = 1 if slice.step is None else slice.step
  #       size = (end - start) * np.prod(shape[slice(i + 1)])
  #       checkSize(size)
  #       array.append(np.arange(start, end, dtype="i4"))
  #       for j in range(i, len(shape)):
  #         array.append(np.arange(0, shape[j], dtype="i4"))
  #     elif isinstance(t, int):
  #       size = np.prod(shape[slice(i + 1)])
  #       checkSize(size)
  #       array.append(t)
  #       for j in range(i, len(shape)):
  #         array.append(np.arange(0, shape[j], dtype="i4"))
  #     elif isinstance(t, list) or isinstance(t, np.ndarray):
  #       t = np.array(t)
  #       checkSize(t.size)
  #       array.append(t)
  #       if (len(t) > shape[i]): raise ValueError("Length of exceeds limit")
  #       for j in range(i, len(shape)):
  #         array.append(np.arange(0, shape[j], dtype="i4"))
  #     else:
  #       raise TypeError("Invalid argument type.")
  # else:
  #   raise TypeError("Invalid argument type.")

  # np.mgrid[0:10, 0:10, 0:10]
  

def getPartitions(indices, dataShape,masterShape):
  """
  TODO: Change description
  :param indices:
  :param dataShape:
  :param masterShape:
  :return:
  """
  masterIndices = getMasterIndices(indices, dataShape,masterShape)
  n = len(dataShape)
  allPartitions=masterIndices[:, :n]
  uniquePartitions,indexPartitions = np.unique(allPartitions,axis=0,return_inverse=True)
  indexDataFile = masterIndices[:, n:]
  indexData = np.concatenate((indexPartitions[:, None], indexDataFile), axis=1)
  return uniquePartitions,indexPartitions,indexData

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