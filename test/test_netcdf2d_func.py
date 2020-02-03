import pytest
import numpy as np
from s3netcdf.netcdf2d_func import createNetCDF,\
  createVariables,getChildShape,getMasterShape,parseDescritor,\
  getIndices,getMasterIndices,getPartitions,dataWrapper

shape1 = [3, 7]
shape2a = [8, 32768]  # 1MB
shape2b = [8, 32769]  # >1MB
shape3a = [1, 262144] # 1MB
shape3b = [1, 262145] # >1MB
shape4a = [8, 8,8,512]  # 1MB
shape4b = [9, 8,8,512]  # >1MB
shape4c = [8, 8,8,513]  # >1MB
shape5a = [16, 32768]  # 2MB
shape5b = [16, 32769]  # >2MB
shape6a = [4, 32768]  # 1MB,type=f8
shape6b = [4, 32769]  # >1MB,type=f8  

def test_getChildShape():
  np.testing.assert_array_equal(getChildShape(shape1), shape1)
  np.testing.assert_array_equal(getChildShape(shape2a), shape2a)
  np.testing.assert_array_equal(getChildShape(shape2b), [4,32769])
  np.testing.assert_array_equal(getChildShape(shape3a), shape3a)
  np.testing.assert_array_equal(getChildShape(shape3b), [1,131073])
  np.testing.assert_array_equal(getChildShape(shape4a), shape4a)
  np.testing.assert_array_equal(getChildShape(shape4b), [5,8,8,512])
  np.testing.assert_array_equal(getChildShape(shape4c), [4,8,8,513])
  np.testing.assert_array_equal(getChildShape(shape5a,size=2), shape5a)
  np.testing.assert_array_equal(getChildShape(shape5b,size=2), [8, 32769])
  np.testing.assert_array_equal(getChildShape(shape6a,dtype="f8"), shape6a)
  np.testing.assert_array_equal(getChildShape(shape6b,dtype="f8"), [2, 32769])
  
  
def test_getMasterShape():
  np.testing.assert_array_equal(getMasterShape(shape1), [1,1,3,7])
  np.testing.assert_array_equal(getMasterShape(shape2a), [1, 1, 8, 32768])
  np.testing.assert_array_equal(getMasterShape(shape2b), [2, 1, 4, 32769])
  np.testing.assert_array_equal(getMasterShape(shape3a), [1, 1, 1, 262144])
  np.testing.assert_array_equal(getMasterShape(shape3b), [1, 2, 1, 131073])
  np.testing.assert_array_equal(getMasterShape(shape4a), [1,1,1,1,8,8,8,512])
  np.testing.assert_array_equal(getMasterShape(shape4b), [2,1,1,1,5,8,8,512])
  np.testing.assert_array_equal(getMasterShape(shape4c), [2,1,1,1,4,8,8,513])
  np.testing.assert_array_equal(getMasterShape(shape5a,size=2), [1, 1, 16, 32768])
  np.testing.assert_array_equal(getMasterShape(shape5b,size=2), [2, 1, 8, 32769])
  np.testing.assert_array_equal(getMasterShape(shape6a,dtype="f8"), [1, 1, 4, 32768])
  np.testing.assert_array_equal(getMasterShape(shape6b,dtype="f8"), [2, 1, 2, 32769])
  
  
def test_parseDescritor():
  np.testing.assert_array_equal(parseDescritor((0),shape1), np.array([0],dtype="int32"))
  np.testing.assert_array_equal(parseDescritor((slice(None,None,None)),shape1), np.arange(3,dtype="int32"))
  np.testing.assert_array_equal(parseDescritor(([0,1,2]),shape1), np.array([0,1,2],dtype="int32"))
  np.testing.assert_array_equal(parseDescritor((0,0),shape1), np.array([[0],[0]],dtype="int32"))
  np.testing.assert_array_equal(parseDescritor((slice(None,None,None),slice(None,None,None)),shape1)[0], np.array([np.arange(3,dtype="int32"),np.arange(7,dtype="int32")])[0])
  np.testing.assert_array_equal(parseDescritor((slice(None,None,None),slice(None,None,None)),shape1)[1], np.array([np.arange(3,dtype="int32"),np.arange(7,dtype="int32")])[1])
  np.testing.assert_array_equal(parseDescritor(([0,1,2],[0,1,2]),shape1), np.array([[0,1,2],[0,1,2]],dtype="int32"))
  np.testing.assert_array_equal(parseDescritor((0,0,0,0),shape4a), np.array([[0],[0],[0],[0]],dtype="int32"))
  
  # Exemptions
  with pytest.raises(Exception) as excinfo1:parseDescritor((0,(0,1)),shape1)
  with pytest.raises(Exception) as excinfo2:parseDescritor(('a'),shape1)
  with pytest.raises(Exception) as excinfo3:parseDescritor((3.0),shape1)
  assert str(excinfo1.value) == 'Invalid argument type.'
  assert str(excinfo2.value) == 'Invalid argument type.'
  assert str(excinfo3.value) == 'Invalid argument type.'
  
  
def test_getIndices():
  np.testing.assert_array_equal(getIndices((0),shape1)[0], np.array([0],dtype="int32"))
  np.testing.assert_array_equal(getIndices((0),shape1)[1], np.arange(7,dtype="int32"))
  
  np.testing.assert_array_equal(getIndices((slice(None,None,None)),shape1)[0], np.arange(3,dtype="int32"))
  np.testing.assert_array_equal(getIndices((slice(None,None,None)),shape1)[1], np.arange(7,dtype="int32"))
  
  np.testing.assert_array_equal(getIndices(([0,1,2]),shape1)[0], np.array([0,1,2],dtype="int32"))
  np.testing.assert_array_equal(getIndices(([0,1,2]),shape1)[1], np.arange(7,dtype="int32"))
  
  np.testing.assert_array_equal(getIndices((0,0),shape1), np.array([[0],[0]],dtype="int32"))
  
  np.testing.assert_array_equal(getIndices((slice(None,None,None),slice(None,None,None)),shape1)[0], np.array([np.arange(3,dtype="int32"),np.arange(7,dtype="int32")])[0])
  np.testing.assert_array_equal(getIndices((slice(None,None,None),slice(None,None,None)),shape1)[1], np.array([np.arange(3,dtype="int32"),np.arange(7,dtype="int32")])[1])
  
  np.testing.assert_array_equal(getIndices(([0,1,2],[0,1,2]),shape1), np.array([[0,1,2],[0,1,2]],dtype="int32"))
  np.testing.assert_array_equal(getIndices((0,0,0,0),shape4a), np.array([[0],[0],[0],[0]],dtype="int32"))
  
  
def test_getMasterIndices():
  # Shape2a
  master2a = getMasterShape(shape2a)
  indices2a_1 = getIndices((0),shape2a)
  indices2a_2 = getIndices((0,0),shape2a)
  indices2a_3 = getIndices((slice(None,None,None),0),shape2a)
  np.testing.assert_array_equal(getMasterIndices(indices2a_1,shape2a,master2a).shape, (32768,4))
  np.testing.assert_array_equal(getMasterIndices(indices2a_2,shape2a,master2a).shape, (1,4))
  np.testing.assert_array_equal(getMasterIndices(indices2a_3,shape2a,master2a).shape, (8,4))
  
  # TODO
  # Specific values

def test_getPartitions():
  # Shape2a
  master2a = getMasterShape(shape2a)
  indices2a_1 = getIndices((0),shape2a)
  indices2a_2 = getIndices((0,0),shape2a)
  indices2a_3 = getIndices((slice(None,None,None),0),shape2a)
  np.testing.assert_array_equal(getPartitions(indices2a_1,shape2a,master2a), np.array([[0,0]],dtype="int32"))
  np.testing.assert_array_equal(getPartitions(indices2a_2,shape2a,master2a), np.array([[0,0]],dtype="int32"))
  np.testing.assert_array_equal(getPartitions(indices2a_3,shape2a,master2a), np.array([[0,0]],dtype="int32"))
  
  # Shape2b
  master2b = getMasterShape(shape2b)
  indices2b_1 = getIndices((0),shape2b)
  indices2b_2 = getIndices((0,0),shape2b)
  indices2b_3 = getIndices((slice(None,None,None),0),shape2b)
  np.testing.assert_array_equal(getPartitions(indices2b_1,shape2b,master2b), np.array([[0,0]],dtype="int32"))
  np.testing.assert_array_equal(getPartitions(indices2b_2,shape2b,master2b), np.array([[0,0]],dtype="int32"))
  np.testing.assert_array_equal(getPartitions(indices2b_3,shape2b,master2b), np.array([[0,0],[1,0]],dtype="int32"))
  
  

def test_dataWrapper():
  def f(part,idata,ipart):
    np.testing.assert_array_equal(part,np.array([0,0],dtype="int32"))
    np.testing.assert_array_equal(idata.shape,(32769))
    np.testing.assert_array_equal(ipart.shape,(32769,2))
  
  master2b = getMasterShape(shape2b)  
  dataWrapper((0),shape2b,master2b,f)
  None

def test_createNetCDF():
  
  None

if __name__ == "__main__":
  test_getChildShape()
  test_getMasterShape()
  test_parseDescritor()
  test_getIndices()
  test_getMasterIndices()
  test_getPartitions()
  test_dataWrapper()
  # test_createNetCDF()