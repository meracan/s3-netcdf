def getFileShape(data, typeSize=8, maxSize=4e+3):
  '''
  Find new array shape based on maximum file size
  Last dimensions gets priority
  TODO change typeSize based on np.int, np.f8, etc...
  :param data: numpy.array
  :param typeSize: int, bytes
  :param maxSize: int,bytes
  :return: numpy.array
  '''
  
  
  dataShape = data.shape
  fileSize = 1
  fileShape = np.ones(len(dataShape), dtype=np.int)
  for i in range(len(dataShape) - 1, -1, -1):
    n = dataShape[i]
    fileSize *= n * typeSize
    p = np.int(np.ceil(fileSize / maxSize))
    if (p > 1):
      fileShape[i] = np.int(np.ceil(n * 1.0 / p))
      break
    else:
      fileShape[i] = n
  
  return fileShape


def getMasterShape(data):
  '''
  Get array shape based on new file partitions
  i.e (10,10)=> (10,1,1,10); 10 partitions on the first axis with file shape of (1,10)
  i.e (10,10)=> (10,2,1,5); 10 partitions on the first axis, 2 partions on the second axis with file shape of (1,5)
  :param data: numpy.array
  :return: numpy.array (shape)
  '''
  
  dataShape = data.shape
  fileShape = getFileShape(data)
  partitions = np.ceil(np.array(dataShape) / newshape).astype('int')
  masterShape = np.insert(fileShape, 0, partitions)
  return masterShape

def getMasterIndices(indices, dataShape,masterShape):
  index = np.ravel_multi_index(indices, dataShape)
  return np.unravel_index(index, masterShape)