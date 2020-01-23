import pytest
from s3netcdf.createNetCDF import NetCDF2D
import numpy as np
svars=[
  dict(name="b",units="m",standard_name="",long_name=""),
]
vars=[
  dict(name="u",units="m/s",standard_name="",long_name=""),
  dict(name="v",units="m/s",standard_name="",long_name=""),
  dict(name="w",units="m/s",standard_name="",long_name=""),
  
]

def test_create():
  nnode=100000
  nelem=100000
 
  lat=np.arange(0, nnode, dtype=np.float64)
  lng=np.arange(0, nnode, dtype=np.float64)-100.0
  elem=np.repeat(np.array([[0,1,2]]),nelem,axis=0)

  netcdf2d=NetCDF2D("mytest")
  netcdf2d.createNC(nnode=nnode,nelem=nelem,vars=svars)
  netcdf2d.createNCA(nnode=nnode, vars=vars, ntime=1, nspectra=50, nfreq=33, ndir=36)
  netcdf2d.write2D("lat",lat)
  netcdf2d.write2D("lng", lng)
  netcdf2d.write2D("elem", elem)
  
  
def test_read():
  None

def test_write():
  netcdf2d = NetCDF2D("mytest")
  netcdf2d.writeOutput("u", [[0,0]],[[0]])
  
if __name__ == "__main__":
  # test_create()
  test_write()
  
  