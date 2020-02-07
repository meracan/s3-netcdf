import pytest
import numpy as np
from input import Intput
from input2 import Intput as Input2
from s3netcdf import NetCDF2D

'''
  Test1 : No partitions
'''

def test_NetCDF2D():
  '''
  
  
  '''
  
  netcdf2d=NetCDF2D(Intput)
  
  #TODO: Compare basic attritbutes
  # summary=netcdf2d.getSummary()
  
  # Get variable shape
  np.testing.assert_array_equal(netcdf2d.getVShape(None,"b"), [3000])
  np.testing.assert_array_equal(netcdf2d.getVShape("s","u"), [720,3000])
  
  #-----------------------------------------------------------------------------
  # Writing / Reading
  #
  # Static variables
  bshape = netcdf2d.getVShape(None,"b")
  bvalue = np.arange(np.prod(bshape)).reshape(bshape)
  netcdf2d[None,"b"] = bvalue
  np.testing.assert_array_equal(netcdf2d[None,"b"], bvalue)
  np.testing.assert_array_equal(netcdf2d[None,"b",0], [0.])
  np.testing.assert_array_equal(netcdf2d[None,"b",[0,10]], [0.,10.])
  np.testing.assert_array_equal(netcdf2d[None,"b",slice(0,10)], np.arange(10))
  
  # Temporal variables
  ushape = netcdf2d.getVShape("s","u")
  print(ushape)
  uvalue = np.arange(np.prod(ushape)).reshape(ushape)
  # netcdf2d["s","u"] = uvalue
  print(netcdf2d["s","u",0:3,[100,200,300,400]])
  # np.testing.assert_array_equal(netcdf2d["s","u"], uvalue)
  
  
  # Reading
  None

def test_NetCDF2D_v2():
  netcdf2d=NetCDF2D(Input2)
  ushape = netcdf2d.getVShape("s","u")
  uvalue = np.arange(np.prod(ushape)).reshape(ushape)
  netcdf2d["s","u"] = uvalue
  np.testing.assert_array_equal(netcdf2d["s","u"], uvalue)
  np.testing.assert_array_equal(netcdf2d["s","u",0], uvalue[0])
  np.testing.assert_array_equal(netcdf2d["s","u",0,:], uvalue[0,:])
  np.testing.assert_array_equal(netcdf2d["s","u",:,0], uvalue[:,0])
  np.testing.assert_array_equal(netcdf2d["s","u",0,0], uvalue[0,0])
  np.testing.assert_array_equal(netcdf2d["s","u",0,131073], uvalue[0,131073])
  np.testing.assert_array_equal(netcdf2d["s","u",0,262144], uvalue[0,262144])
  np.testing.assert_array_equal(netcdf2d["s","u",1,100], uvalue[1,100])
  np.testing.assert_array_equal(netcdf2d["s","u",1,131073], uvalue[1,131073])
  np.testing.assert_array_equal(netcdf2d["s","u",1,262144], uvalue[1,262144])
  
  np.testing.assert_array_equal(netcdf2d["s","u",0:2,0], uvalue[0:2,0])
  np.testing.assert_array_equal(netcdf2d["s","u",0:2,131073], uvalue[0:2,131073])
  np.testing.assert_array_equal(netcdf2d["s","u",0:2,262144], uvalue[0:2,262144])
  np.testing.assert_array_equal(netcdf2d["s","u",[0,1],100], uvalue[[0,1],100])
  np.testing.assert_array_equal(netcdf2d["s","u",[0,1],131073], uvalue[[0,1],131073])
  np.testing.assert_array_equal(netcdf2d["s","u",[0,1],262144], uvalue[[0,1],262144])
  
  np.testing.assert_array_equal(netcdf2d["s","u",0:2,10:20], uvalue[0:2,10:20])
  np.testing.assert_array_equal(netcdf2d["s","u",0:2,[0,262144]], uvalue[0:2,[0,262144]])
  np.testing.assert_array_equal(netcdf2d["s","u",1:2,131060:262144], np.squeeze(uvalue[1:2,131060:262144])) # For some reason, numpy does not squeeze this
  np.testing.assert_array_equal(netcdf2d["s","u",1,131060:262144], uvalue[1,131060:262144])
  np.testing.assert_array_equal(netcdf2d["s","u",0:2,[0,131060,131076,262144]], uvalue[0:2,[0,131060,131076,262144]])
  
  z10=np.zeros(10)
  netcdf2d["s","u",0,0:10] = z10
  netcdf2d["s","u",0,131073:131083] = z10
  netcdf2d["s","u",0:2,100:110] = np.repeat(z10,2)
  np.testing.assert_array_equal(netcdf2d["s","u",0,0:10], z10)
  np.testing.assert_array_equal(netcdf2d["s","u",0,131073:131083], z10)
  np.testing.assert_array_equal(netcdf2d["s","u",0,100:110], z10)
  np.testing.assert_array_equal(netcdf2d["s","u",1,100:110], z10)
  
  
  eshape = netcdf2d.getVShape("ss","e")
  evalue = np.arange(np.prod(eshape)).reshape(eshape)
  # netcdf2d["ss","e"] = evalue
  # np.testing.assert_array_equal(netcdf2d["ss","e"], np.squeeze(evalue))
  np.testing.assert_array_equal(netcdf2d["ss","e",0,[0,100,200],0:10], np.squeeze(evalue[0,[0,100,200],0:10]))
  # print(netcdf2d["ss","e",0,0].shape)
  # print(netcdf2d["ss","e"])
  # print(evalue)
  
  
def test_NetCDF2D_exemptions():
  None
  

  
if __name__ == "__main__":
  # test_NetCDF2D()
  test_NetCDF2D_v2()
  # test_NetCDF2D_exemptions()