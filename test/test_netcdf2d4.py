import pytest
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta

Input = dict(
  name="input4",
  cacheLocation=r"../s3",
  localOnly=True,
  
  bucket="uvic-bcwave",
  cacheSize=10.0,
  ncSize=10.0,
  
  nca = dict(
    metadata=dict(title="Input2"),
    dimensions = dict(
      npe=3,
      nelem=262145,
      nnode=262145,
      ntime=120,
      nspectra=300,
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
      nodes=dict(dimensions=["nnode"],variables=dict(
        bed=dict(type="f4",units="m" ,standard_name="" ,long_name=""),
        friction=dict(type="f4" ,units="" ,standard_name="" ,long_name=""),
        )),
      s=dict(dimensions=["ntime", "nnode"] ,variables=dict(
        a=dict(type="f8",units="m" ,standard_name="" ,long_name=""),
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
  
  
  # netcdf2d["s","a",0] = value[0]
  # np.testing.assert_array_equal(netcdf2d["s","a",0], np.squeeze(value[0]))
  
  # netcdf2d["s","a",50] = value[50]
  # np.testing.assert_array_equal(netcdf2d["s","a",50], np.squeeze(value[50]))
  
  # netcdf2d["s","a",10:40] = value[10:40]
  # np.testing.assert_array_equal(netcdf2d["s","a",10:40], np.squeeze(value[10:40]))
  
  # netcdf2d["s","a",100:101] = value[100:101]
  # np.testing.assert_array_equal(netcdf2d["s","a",100:101], np.squeeze(value[100:101]))  

  netcdf2d["s","a"] = value
  # np.testing.assert_array_equal(netcdf2d["s","a"], np.squeeze(value))
  
  netcdf2d.cache.delete()

if __name__ == "__main__":
  test_NetCDF2D_4()