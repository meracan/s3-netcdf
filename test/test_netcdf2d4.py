import pytest
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta

Input = dict(
  name="input4",
  cacheLocation=r"../s3",
  localOnly=True,
  squeeze=True,
  bucket="uvic-bcwave",
  cacheSize=10.0,
  ncSize=10.0,
  
  nca = dict(
    metadata=dict(title="Input2"),
    dimensions = dict(
      npe=3,
      nelem=262145,
      nnode=262145,
      ntime=110,
      nspectra=300,
      nstation=100,
      nsnode=20,
      nfreq=33,
      ndir=36,
    ),
    groups=dict(
      elem=dict(dimensions=["nelem","npe"],variables=dict(
        elem=dict(type="i4", units="" ,standard_name="" ,long_name=""),
      )),
      time=dict(dimensions=["ntime"],variables=dict(
        time=dict(type="f8",units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="" ,long_name=""),
        )),
      node=dict(dimensions=["nnode"],variables=dict(
        bed=dict(type="f4",units="m" ,standard_name="" ,long_name=""),
        friction=dict(type="f4" ,units="" ,standard_name="" ,long_name=""),
        )),
      s=dict(dimensions=["ntime", "nnode"] ,variables=dict(
        a=dict(type="f8",units="m" ,standard_name="" ,long_name=""),
        )),
      spc=dict(dimensions=["nstation", "nsnode","ntime","nfreq","ndir"] ,variables=dict(
        energy=dict(type="f8",units="m" ,standard_name="" ,long_name=""),
        )),
    )
  )
)

def test_NetCDF2D_4():
  """
  Advanced testing to test index assignment
  ----------
  """
  netcdf2d=NetCDF2D(Input)
  elemshape = netcdf2d.groups["s"].shape
  
  value = np.arange(np.prod(elemshape)).reshape(elemshape)
  
  
  netcdf2d["s","a",0] = value[0]
  np.testing.assert_array_equal(netcdf2d["s","a",0], value[0])
  
  netcdf2d["s","a",50] = value[50]
  np.testing.assert_array_equal(netcdf2d["s","a",50], value[50])
  
  netcdf2d["s","a",10:40] = value[10:40]
  np.testing.assert_array_equal(netcdf2d["s","a",10:40], value[10:40])
  
  netcdf2d["s","a",100:101] = value[100:101]
  np.testing.assert_array_equal(netcdf2d["s","a",100:101],np.squeeze(value[100:101]))  

  netcdf2d["s","a"] = value
  np.testing.assert_array_equal(netcdf2d["s","a"], value)
  
  value2 = np.arange(20*110*33*36).reshape((1,20,110,33,36))
  
  netcdf2d["spc","energy",0,:] = value2
  np.testing.assert_array_equal(netcdf2d["spc","energy",0,0:20], np.squeeze(value2))
  np.testing.assert_array_equal(netcdf2d["spc","energy",0], np.squeeze(value2))
  
  shape=netcdf2d.groups['s'].shape
  child=netcdf2d.groups['s'].child
  n=shape[0]
  step=child[0]
  
  
  for i in range(0,n,step):
    j=np.minimum(n,i+step)
    netcdf2d["s","a",i:j] = np.ones(child)
  np.testing.assert_array_equal(netcdf2d["s","a",0], np.ones((262145)))
  
  
  
  netcdf2d.cache.delete()

# def test_NetCDF2D_4_quick():
  
#   netcdf2d=NetCDF2D(Input)
#   bedshape = netcdf2d.groups["node"].shape
#   netcdf2d["s","a",0] = np.ones(list(bedshape)[0])
  
  
  
#   netcdf2d.cache.delete()

if __name__ == "__main__":
  test_NetCDF2D_4()
  # test_NetCDF2D_4_quick()