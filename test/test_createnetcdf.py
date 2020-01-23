import pytest
from s3netcdf.createNetCDF import NetCDF2D
import numpy as np

from test.input import intput1




def test_create():
  netcdf2d=NetCDF2D(**intput1)
  # netcdf2d.write("u", [[0, 0],[1, 1],[1, 2]], [0,1,2], "s")
  
  print((netcdf2d.write("u","s"))[[0,1],1])
  # netcdf2d.write("lat", [[0, 0]], [0], "s")
  
def test_write():
  netcdf2d = NetCDF2D(**intput1)
  netcdf2d.write("u", [[0, 0]], [0], "s")
  netcdf2d.write("u", [[0, 0]], [0], "s")
  netcdf2d.write("u",[[0,0]],[0],"s")

  #
  # lat=np.arange(0, nnode, dtype=np.float64)
  # lng=np.arange(0, nnode, dtype=np.float64)-100.0
  # elem=np.repeat(np.array([[0,1,2]]),nelem,axis=0)
  
def test_read():
  netcdf2d = NetCDF2D(**intput1)
  
  # netcdf2d.findVariable("e")
  


# def test_write():
#   netcdf2d = NetCDF2D("mytest")
#   netcdf2d.writeOutput("u", [[0,0]],[[0]])
  
if __name__ == "__main__":
  test_create()
  # test_read()
  #
  # test_write()
  