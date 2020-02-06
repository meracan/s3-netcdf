import pytest
import numpy as np
from input import Intput
from input2 import Intput as Input2
from s3netcdf import NetCDF2D

def test_NetCDF2D():
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
  # print(ushape)
  uvalue = np.arange(np.prod(ushape)).reshape(ushape)
  netcdf2d["s","u",:2,[0,200]] = np.arange(4)
  print(netcdf2d["s","u",0,:201])

  
def test_NetCDF2D_exemptions():
  None
  

  
if __name__ == "__main__":
  # test_NetCDF2D()
  test_NetCDF2D_v2()
  # test_NetCDF2D_exemptions()