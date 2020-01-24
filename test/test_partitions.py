import pytest
from s3netcdf.partitions import getChildShape,\
  getMasterShape,getMasterIndices,getPartitions,concatenatePartitions,indexMulti,\
  createIndices
  # getMasterSlices,getPartitionsSlices,

import numpy as np

shape1 = [3, 7]
shape2 = [8, 16384]  # 1MB
shape3 = [8, 16385]  # >1MB
# shape4 = [2, 131073]  # >1MB on last column
shape4 = [24*365*15, 300000]  # >1MB on last column

data1 = np.reshape(np.arange(shape1[0] * shape1[1]), shape1)
data2 = np.reshape(np.arange(shape2[0] * shape2[1], dtype=np.float64), shape2)
data3 = np.reshape(np.arange(shape3[0] * shape3[1], dtype=np.float64), shape3)
# data4 = np.reshape(np.arange(shape4[0] * shape4[1], dtype=np.float64), shape4)

def test_getChildShape():
  np.testing.assert_array_equal(getChildShape(shape1), shape1)
  np.testing.assert_array_equal(getChildShape(shape2), shape2)
  np.testing.assert_array_equal(getChildShape(shape3), [4,16385])
  np.testing.assert_array_equal(getChildShape(shape3,maxSize=2), shape3)
  np.testing.assert_array_equal(getChildShape(shape4), [1, 65537])

def test_getMasterShape():
  np.testing.assert_array_equal(getMasterShape(shape1), [1,1,3,7])
  np.testing.assert_array_equal(getMasterShape(shape2), [1, 1, 8, 16384])
  np.testing.assert_array_equal(getMasterShape(shape3), [2, 1, 4, 16385])
  np.testing.assert_array_equal(getMasterShape(shape4), [2, 2, 1, 65537])

def test_getMasterIndices():
  np.testing.assert_array_equal(getMasterIndices([1, 0], shape1, getMasterShape(shape1)), [[0, 0, 1, 0]])
  np.testing.assert_array_equal(getMasterIndices([1, 0],shape1,getMasterShape(shape1)), [[0, 0, 1, 0]])
  np.testing.assert_array_equal(getMasterIndices([[0,0],[0, 1],[1, 0]], shape1, getMasterShape(shape1)), [[0, 0, 0, 0],[0, 0,0 , 1],[0, 0, 1, 0]])
  np.testing.assert_array_equal(getMasterIndices([1, 0], shape3, getMasterShape(shape3)), [[0, 0, 1, 0]])
  np.testing.assert_array_equal(getMasterIndices([6, 0], shape3, getMasterShape(shape3)), [[1, 0, 2, 0]])
  np.testing.assert_array_equal(getMasterIndices([1, 0], shape4, getMasterShape(shape4)), [[0, 1, 0, 65536]])
  np.testing.assert_array_equal(getMasterIndices([1, 1], shape4, getMasterShape(shape4)), [[1, 0, 0, 0]])

def test_createIndices():
  indices = createIndices((2,100),(0))
  
  # np.mgrid[0:10, 0:10, 0:10]
  None
  
def test_getPartitions():
  # uniquePartitions,indexPartitions,indexData = getPartitions([0, 0], shape4, getMasterShape(shape4))
  # np.testing.assert_array_equal(uniquePartitions, [[0, 0]])
  # np.testing.assert_array_equal(indexPartitions, [0])
  # np.testing.assert_array_equal(indexData, [[0,0,0]])
  
  array =np.concatenate(np.indices((2,300000)).transpose(1,2,0))
  array[:,0] = array[:,0]+1

  
  uniquePartitions,indexPartitions,indexData = getPartitions(array, shape4, getMasterShape(shape4,maxSize=1))
  print(uniquePartitions,)
  print(uniquePartitions)
  
  # uniquePartitions, indexPartitions,indexData = getPartitions([[0,0],[0, 1],[1, 1]], shape4, getMasterShape(shape4))
  # np.testing.assert_array_equal(uniquePartitions, [[0, 0],[1,0]])
  # np.testing.assert_array_equal(indexPartitions, [0,0,1])
  # np.testing.assert_array_equal(indexData, [[0,0,0],[0,0,1],[1,0,0]])


def test_getMasterSlices():
  np.testing.assert_array_equal(getMasterSlices((slice(0, 3, None), slice(0, 2, None)), shape4, getMasterShape(shape4)), [[0, 0, 0, 0], [1, 0, 0, 1]])

def test_getPartitionsSlices():
  partitions = getPartitionsSlices((slice(0, 1, None), slice(0, 10, None)),shape4,getMasterShape(shape4))
  print(partitions)
  


def test_indexMulti():
  with pytest.raises(Exception) as excinfo:
    indexMulti([0],[0])
  assert str(excinfo.value) == 'Indices shape needs to be 2D'
  
  with pytest.raises(Exception) as excinfo2:
    indexMulti([[0,0]],[[0]])
  assert str(excinfo2.value) == 'Shapes are not identical'
  
  # Test 0, 1D
  np.testing.assert_array_equal( indexMulti([0], [[0]]), [0])

  # Test 0b, 2D
  np.testing.assert_array_equal(indexMulti([[0]], [[0,0]]), [0])
  
  
  # Test 1, 3D
  dataShape=[2,2,2] # [# of files,x,y]
  n= np.prod(dataShape)
  data = np.arange(n,dtype="f8").reshape(dataShape)
  
  indices = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1, 1, 0],
    [1, 1, 1]
  ],dtype="int")
  np.testing.assert_array_equal(indexMulti(data,indices), np.arange(n,dtype="f8"))

  dataShape = [2, 1, 1, 2, 2]  # [# of files,x,y,z,w]
  n = np.prod(dataShape)
  data = np.arange(n, dtype="f8").reshape(dataShape)

  # Test 2, 5D
  indices = np.array([  # , [fileId,x,y,z,w]]
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],

  ], dtype="int")
  np.testing.assert_array_equal(indexMulti(data, indices), np.arange(n, dtype="f8"))
  
  
def test_concatenatePartitions():
  fileShape = getChildShape(data4.shape)
  n=np.prod(fileShape)
  
  searchArray=[[0, 0], [0, 1], [1, 1]]
  
  masterShape=getMasterShape(data4.shape)
  uniquePartitions, indexPartitions,indexData = getPartitions(searchArray, shape4, masterShape)
  
  # Get Data part
  file1=(np.arange(n, dtype="f8") / n + 0).reshape(fileShape)
  file2 = (np.arange(n, dtype="f8") / n + 1).reshape(fileShape)
  file3 = (np.arange(n, dtype="f8") / n + 2).reshape(fileShape)
  file4 = (np.arange(n, dtype="f8") / n + 3).reshape(fileShape)
  files=np.array([[file1,file2],[file3,file4]])
  
  marray= []
  #
  for i,id in enumerate(uniquePartitions):
    marray.append(files[id[0],id[1]])
  # End of get Data part
  
  array = concatenatePartitions(marray)
  results = indexMulti(array,indexData)
  
  np.testing.assert_array_equal(results, [0,1/n,2])
  
  
  
  


if __name__ == "__main__":
  test_createIndices()
  # test_getPartitions()
  # test_getChildShape()
  # test_getMasterShape()
  # test_getMasterIndices()
  # test_getMasterSlices()
  # test_getPartitionsSlices()
  
  # test_getPartitions()
  # test_indexMulti()
  # test_concatenatePartitions()
  # a=np.array([[1,0],[5,0]]).T
  # print(a.shape)
  # index = np.ravel_multi_index(np.array([[1,0],[4,0]]).T, (5,2))
  # for i in np.arange(index[0],index[1]+1):
  #   if(i==index[0]):
  #     # Start
  #     None
  #   elif (i == index[1]+1):
  #     # Start
  #     None
  #   else:
  #     None
  #   print(np.array(np.unravel_index(i, (5,2))).T)
  #