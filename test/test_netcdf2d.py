import pytest
from s3netcdf.createNetCDF import NetCDF2D
import numpy as np

from test.input import intput1




def test_create():
  netcdf2d=NetCDF2D(**intput1)
  # netcdf2d["s","v",0:10,[0,1]]=np.arange(1000000,dtype="f4").reshape(10,100000)
  # netcdf2d["s","v",0:720,[0,299999]]=np.random.rand(216000000) #np.arange(216000000,dtype="f4").reshape(720,300000)*np.rand()
  print(netcdf2d["s","v",10,1])
  
  # netcdf2d.write("u", [[0, 0],[1, 1],[1, 2]], [0,1,2], "s")
  
  # index = np.ravel_multi_index([:,:], (10,10))
  # print((netcdf2d.write("u","s"))[0,0])
  # print((netcdf2d.write("u","s"))[[0,1],:,100])
  # netcdf2d.write("lat", [[0, 0]], [0], "s")
  
def test_write():
  netcdf2d = NetCDF2D(**intput1)
  # netcdf2d.write("u", [[0, 0]], [0], "s")
  # netcdf2d.write("u", [[0, 0]], [0], "s")
  # netcdf2d.write("u",[[0,0]],[0],"s")

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
  