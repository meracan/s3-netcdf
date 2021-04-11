import os
import pytest
import numpy as np
from netCDF4 import Dataset
from netcdf import NetCDF
from s3netcdf.s3netcdf_func import createNetCDF,\
  getChildShape,getMasterShape,parseDescriptor,\
  getIndices,getMasterIndices,getPartitions,parseIndex,isQuickSet,isQuickGet

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
shape7 = [6049,764300]


def test_getChildShape():
  np.testing.assert_array_equal(getChildShape(shape1), shape1)
  np.testing.assert_array_equal(getChildShape(shape2a), shape2a)
  np.testing.assert_array_equal(getChildShape(shape2b), [4,32769])
  np.testing.assert_array_equal(getChildShape(shape3a), shape3a)
  np.testing.assert_array_equal(getChildShape(shape3b), [1,131073])
  np.testing.assert_array_equal(getChildShape(shape4a), shape4a)
  np.testing.assert_array_equal(getChildShape(shape4b), [5,8,8,512])
  np.testing.assert_array_equal(getChildShape(shape4c), [4,8,8,513])
  np.testing.assert_array_equal(getChildShape(shape5a,ncSize=2), shape5a)
  np.testing.assert_array_equal(getChildShape(shape5b,ncSize=2), [8, 32769])
  np.testing.assert_array_equal(getChildShape(shape6a,dtype="f8"), shape6a)
  np.testing.assert_array_equal(getChildShape(shape6b,dtype="f8"), [2, 32769])
  # np.testing.assert_array_equal(getChildShape(shape7,ncSize=10), [8, 32769])
  
  
def test_getMasterShape():
  np.testing.assert_array_equal(getMasterShape(shape1), [1,1,3,7])
  np.testing.assert_array_equal(getMasterShape(shape2a), [1, 1, 8, 32768])
  np.testing.assert_array_equal(getMasterShape(shape2b), [2, 1, 4, 32769])
  np.testing.assert_array_equal(getMasterShape(shape3a), [1, 1, 1, 262144])
  np.testing.assert_array_equal(getMasterShape(shape3b), [1, 2, 1, 131073])
  np.testing.assert_array_equal(getMasterShape(shape4a), [1,1,1,1,8,8,8,512])
  np.testing.assert_array_equal(getMasterShape(shape4b), [2,1,1,1,5,8,8,512])
  np.testing.assert_array_equal(getMasterShape(shape4c), [2,1,1,1,4,8,8,513])
  np.testing.assert_array_equal(getMasterShape(shape5a,ncSize=2), [1, 1, 16, 32768])
  np.testing.assert_array_equal(getMasterShape(shape5b,ncSize=2), [2, 1, 8, 32769])
  np.testing.assert_array_equal(getMasterShape(shape6a,dtype="f8"), [1, 1, 4, 32768])
  np.testing.assert_array_equal(getMasterShape(shape6b,dtype="f8"), [2, 1, 2, 32769])
  
  
def test_parseDescriptor():
  np.testing.assert_array_equal(parseDescriptor((0),shape1), np.array([0],dtype="int32"))
  np.testing.assert_array_equal(parseDescriptor((slice(None,None,None)),shape1), np.arange(3,dtype="int32"))
  np.testing.assert_array_equal(parseDescriptor(([0,1,2]),shape1), np.array([0,1,2],dtype="int32"))
  np.testing.assert_array_equal(parseDescriptor((0,0),shape1), np.array([[0],[0]],dtype="int32"))
  np.testing.assert_array_equal(parseDescriptor((slice(None,None,None),slice(None,None,None)),shape1)[0], np.array([np.arange(3,dtype="int32"),np.arange(7,dtype="int32")])[0])
  np.testing.assert_array_equal(parseDescriptor((slice(None,None,None),slice(None,None,None)),shape1)[1], np.array([np.arange(3,dtype="int32"),np.arange(7,dtype="int32")])[1])
  np.testing.assert_array_equal(parseDescriptor(([0,1,2],[0,1,2]),shape1), np.array([[0,1,2],[0,1,2]],dtype="int32"))
  np.testing.assert_array_equal(parseDescriptor((0,0,0,0),shape4a), np.array([[0],[0],[0],[0]],dtype="int32"))
  
  # Exemptions
  with pytest.raises(Exception) as excinfo1:parseDescriptor((0,(0,1)),shape1)
  with pytest.raises(Exception) as excinfo2:parseDescriptor(('a'),shape1)
  with pytest.raises(Exception) as excinfo3:parseDescriptor((3.0),shape1)
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
  
  # TODO: Shape3a and Shape3b
  
  

def test_createNetCDF():
  
  folder = "../s3"
  filePath = os.path.join(folder,"test1.nc")
  metadata=dict(title="Mytitle")
  dimensions = dict(
    npe=3,
    nnode=100,
    ntime=1000,
    nelem=10
  )
  variables=dict(
    a=dict(type="float32" ,dimensions=["nnode"] ,units="m" ,standard_name="" ,long_name="",least_significant_digit=3),
    lat=dict(type="float64" ,dimensions=["nnode"] ,units="m" ,standard_name="" ,long_name=""),
    lng=dict(type="float64" ,dimensions=["nnode"] ,units="m" ,standard_name="" ,long_name=""),
    elem=dict(type="int32" ,dimensions=["nelem"] ,units="m" ,standard_name="" ,long_name=""),
    time=dict(type="float64" ,dimensions=["ntime"] ,units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="" ,long_name=""),
  )
  variables2 =dict(
    u=dict(type="float32" ,units="m/s" ,standard_name="" ,long_name=""),
  )
  groups=dict(
    s=dict(dimensions=["ntime", "nnode"] ,variables=variables2),
  )
  
  createNetCDF(filePath,folder=folder,metadata=metadata,dimensions=dimensions,variables=variables,ncSize=1.0)
  with NetCDF(filePath,"r") as nc: 
    np.testing.assert_array_equal(nc.obj['metadata'],metadata)
    np.testing.assert_array_equal(nc.obj['dimensions'],dimensions)
    np.testing.assert_array_equal(nc.obj['variables']['a'],{'dimensions': ['nnode'], 'type': 'f', 'least_significant_digit': 3, 'units': 'm', 'standard_name': '', 'long_name': '', 'ftype': 'f'})
  
  createNetCDF(filePath,folder=folder,metadata=metadata,dimensions=dimensions,groups=groups,ncSize=1.0)
  with NetCDF(filePath,"r") as nc:
    np.testing.assert_array_equal(nc.obj['metadata'],metadata)
    np.testing.assert_array_equal(nc.obj['dimensions'],dimensions)
    np.testing.assert_array_equal(nc.obj['groups']["s"]['variables'],{'u': {'dimensions': ['ntime', 'nnode'], 'type': 'f', 'units': 'm/s', 'standard_name': '', 'long_name': '', 'ftype': 'f'}})
  
def test_parseIndex():
  assert parseIndex("0")==0
  assert parseIndex(0)==0
  assert parseIndex(":")==slice(None,None,None)
  assert parseIndex("1:")==slice(1,None,None)
  assert parseIndex(":10")==slice(None,10,None)
  assert parseIndex("0:10")==slice(0,10,None)
  assert parseIndex("1:10")==slice(1,10,None)
  assert parseIndex("[1,2]")==[1,2]
  
  with pytest.raises(Exception) as excinfo1:parseIndex("a")
  assert str(excinfo1.value) == 'Format needs to be \"{int}\" or \":\" or \"{int}:{int}\" or \"[{int},{int}]\"'

  

if __name__ == "__main__":
  test_getChildShape()
  test_getMasterShape()
  test_parseDescriptor()
  test_getIndices()
  test_getMasterIndices()
  test_getPartitions()
  test_createNetCDF()
  test_parseIndex()