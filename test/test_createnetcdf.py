import pytest
from s3netcdf.createNetCDF import createNetCDF,writeNetCDF

def test_create():
  createNetCDF("test.nca",master=True,ntime=1)
  createNetCDF("test.nc")
  
def test_read():
  None

def test_write():
  writeNetCDF(src_master,indices,data)
  
  None
  
if __name__ == "__main__":
  test_create()