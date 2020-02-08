import shutil
import pytest
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta

Input = dict(
  name="input1",
  cacheLocation=r"../s3",
  localOnly=True,
  autoUpload=True,
  bucket="merac-dev",
  cacheSize=10.0,
  ncSize=1.0,
  nca = dict(
    metadata=dict(title="Input1"),
    dimensions = [
      dict(name="npe" ,value=3),
      dict(name="nelem" ,value=500),
      dict(name="nnode" ,value=1000),
      dict(name="ntime" ,value=2),
    ],
    groups=[
      dict(name="elem",dimensions=["nelem","npe"],variables=[
        dict(name="elem",type="f4", units="" ,standard_name="" ,long_name=""),
        ]),
      dict(name="time",dimensions=["ntime"],variables=[
        dict(name="time",type="f4",units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="" ,long_name=""),
        ]),
      dict(name="nodes",dimensions=["nnode"],variables=[
        dict(name="bed",type="f4",units="m" ,standard_name="" ,long_name=""),
        dict(name="friction",type="f4" ,units="" ,standard_name="" ,long_name=""),
        ]),
      dict(name="s" ,dimensions=["ntime", "nnode"] ,variables=[
        dict(name="a",type="f4",units="m" ,standard_name="" ,long_name=""),
        ]),
      dict(name="t" ,dimensions=["nnode" ,"ntime"] ,variables=[
        dict(name="a",type="f4",units="m" ,standard_name="" ,long_name=""),
        ]),
    ]
  )
)

def test_NetCDF2D_1():
  shutil.rmtree(Input['cacheLocation'])
  # Create and save NetCDF2D object
  netcdf2d=NetCDF2D(Input)
  
  # Get variable shape, create dummy data using arange and save it to netcdf2d:
  #  1 parameter is name of group
  #  2 parameter is name of variable
  elemshape = netcdf2d.getVShape("elem","elem")
  elemvalue = np.arange(np.prod(elemshape)).reshape(elemshape)
  netcdf2d["elem","elem"] = elemvalue
  
  # Read variable and compare with the array above
  np.testing.assert_array_equal(netcdf2d["elem","elem"], elemvalue)
  
  # Write and read "time" variable
  timeshape = netcdf2d.getVShape("time","time")
  timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
  netcdf2d["time","time"] = timevalue
  np.testing.assert_array_equal(netcdf2d["time","time"], timevalue)
  
  # Write and read "bed" variable
  bedshape = netcdf2d.getVShape("nodes","bed")
  bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
  netcdf2d["nodes","bed"] = bedvalue
  np.testing.assert_array_equal(netcdf2d["nodes","bed"], bedvalue)
  
  # Write and read "a" variable
  sashape = netcdf2d.getVShape("s","a")
  savalue = np.arange(np.prod(sashape)).reshape(sashape)
  netcdf2d["s","a"] = savalue
  np.testing.assert_array_equal(netcdf2d["s","a"], savalue)
  
  # Single int and float will be copied based on the index shape
  netcdf2d["s","a",0,100:200] = 0.0
  np.testing.assert_array_equal(netcdf2d["s","a",0,100:200], np.zeros(100))
  
  tashape = netcdf2d.getVShape("t","a")
  tavalue = np.arange(np.prod(tashape)).reshape(tashape)
  netcdf2d["t","a"] = tavalue
  np.testing.assert_array_equal(netcdf2d["t","a"], tavalue)

def test_NetCDF2D_1b():
  # Simple test case to verify upload, clear cache and download from s3
  netcdf2d=NetCDF2D(Input)
  netcdf2d.cache.uploadNCA()
  netcdf2d.cache.uploadNC()
  netcdf2d.cache.clear()
  
  # Write and read "bed" variable
  bedshape = netcdf2d.getVShape("nodes","bed")
  bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
  netcdf2d["nodes","bed"] = bedvalue
  np.testing.assert_array_equal(netcdf2d["nodes","bed"], bedvalue)
  

if __name__ == "__main__":
  test_NetCDF2D_1()
  test_NetCDF2D_1b()