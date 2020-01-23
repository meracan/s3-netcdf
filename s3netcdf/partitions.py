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