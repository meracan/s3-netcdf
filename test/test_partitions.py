import pytest
import timeit
from s3netcdf.partitions import getChildShape,\
  getMasterShape,getMasterIndices,getPartitions,concatenatePartitions,indexMulti,\
  createIndices
  # getMasterSlices,getPartitionsSlices,

import numpy as np

shape1 = [3, 7]
shape2 = [8, 16384]  # 1MB
shape3 = [8, 16385]  # >1MB
shape4 = [2, 131073]  # >1MB on last column
shape5 = [100, 50,5,5]  # >1MB on last column
shape6 = [24*30,300000]


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
  # getMasterIndices((slice(0,10,None)), shape6, getMasterShape(shape6,maxSize=0.01))
  import time
  start = time.time()
  data=np.ones(2999990,dtype=np.float32)
  files=np.zeros(( 10,2,1,150000),dtype=np.float32)
  def f(part,idata,ipart):
    file = files[part[0],part[1]]
    # print(data.shape,idata.shape)
    file[ipart[:,0],ipart[:,1]]=data[idata]
    
    # print(part,idata,ipart)
    # None

  getPartitions((slice(0,10,None),slice(1,None,None)), shape6, getMasterShape(shape6,maxSize=2),f)
  end = time.time()
  print(files[1,0])
  print(end - start)
  # np.testing.assert_array_equal(getMasterIndices((1,0), shape1, getMasterShape(shape1)), [[0, 0, 1, 0]])
  
  # np.testing.assert_array_equal(getMasterIndices((0,slice(0,2)),shape1,getMasterShape(shape1)), [[0, 0, 0, 0],[0, 0, 0, 1]])
  # np.testing.assert_array_equal(getMasterIndices((1,0), shape1, getMasterShape(shape1)), [[0, 0, 0, 0],[0, 0,0 , 1],[0, 0, 1, 0]])
  # np.testing.assert_array_equal(getMasterIndices((1, 0), shape3, getMasterShape(shape3)), [[0, 0, 1, 0]])
  # np.testing.assert_array_equal(getMasterIndices((6, 0), shape3, getMasterShape(shape3)), [[1, 0, 2, 0]])
  # np.testing.assert_array_equal(getMasterIndices((1, 0), shape4, getMasterShape(shape4),getChildShape(shape4)), [[0, 1, 0, 65536]])
  # np.testing.assert_array_equal(getMasterIndices((0), shape5, getMasterShape(shape5,maxSize=0.01),getChildShape(shape5,maxSize=0.01)), [[[[0, 0, 0, 0]]]])

def test_createIndices():
  shape = (2,5,3,4)
  indices = createIndices(shape,(1))
  meshgrid = np.meshgrid(*indices,indexing="ij")
  index = np.ravel_multi_index(meshgrid, shape)
  r = np.array(np.unravel_index(index, (10,1,1,1,1,1,3,4)))
  
  
  # print(np.concatenate(np.meshgrid(*indices,indexing="ij")).transpose(1,2,0).reshape((np.prod(np.array(shape)),3)))
  # print(np.concatenate(np.meshgrid(indices[0],indices[1]),axis=1))
  # print(np.mgrid[[1,2],[1,2]])
  None
  
def test_getPartitions():
  # uniquePartitions,indexPartitions,indexData = getPartitions([0, 0], shape4, getMasterShape(shape4))
  # np.testing.assert_array_equal(uniquePartitions, [[0, 0]])
  # np.testing.assert_array_equal(indexPartitions, [0])
  # np.testing.assert_array_equal(indexData, [[0,0,0]])
  
  # array =np.concatenate(np.indices((2,300000)).transpose(1,2,0))
  array[:,0] = array[:,0]+1

  
  # uniquePartitions,indexPartitions,indexData = getPartitions(array, shape4, getMasterShape(shape4,maxSize=1))
  # print(uniquePartitions,)
  # print(uniquePartitions)
  
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
  # test_createIndices()
  
  # test_getChildShape()
  # test_getMasterShape()
  test_getMasterIndices()
  # test_getPartitions()
  
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