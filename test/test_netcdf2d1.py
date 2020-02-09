import shutil
import pytest
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta

Input = dict(
  name="input1",
  cacheLocation=r"../s3",
  localOnly=True,
  bucket="merac-dev",
  cacheSize=10.0,
  ncSize=1.0,
  nca = dict(
    metadata=dict(title="Input1"),
    dimensions = {"nnode":1000,"ntime":2},
    groups={
      "time":{"dimensions":["ntime"],"variables":{
          "time":{"type":"f8", "units":"hours since 1970-01-01 00:00:00.0","calendar":"gregorian" ,"standard_name":"Datetime" ,"long_name":"Datetime"}
        }
      },
      "nodes":{"dimensions":["nnode"],"variables":{
          "bed":{"type":"f4", "units":"m" ,"standard_name":"Bed Elevation, m" ,"long_name":"Description of data source"},
          "friction":{"type":"f4", "units":"" ,"standard_name":"Bed Friction (Manning's)" ,"long_name":"Description of data source"}
        }
      },
      "s":{"dimensions":["ntime","nnode"],"variables":{
          "a":{"type":"f4", "units":"m" ,"standard_name":"a variable" ,"long_name":"Description of a"}
        }
      },
      "t":{"dimensions":["nnode","ntime"],"variables":{
          "a":{"type":"f4", "units":"m" ,"standard_name":"a variable" ,"long_name":"Description of a"}
        }
      }
    }
  )
)

def test_NetCDF2D_1():
  """
  Basic testing to test all features: create, write & read, caching & s3 commands.
  Logical process:
    localOnly:true
    create (check basic parameters)
    write
    read (check values)
    upload to s3
    delete cache (check non-existing files)
    read (autodownloads from s3 and check values)
    delete cache
    delete s3 (check non-existing files in s3)
    
    localOnly:false
    create (check basic parameters)
    write
    delete cache (check non-existing files)
    read (check values)
    delete cache
    delete s3 (check non-existing files in s3)
    
  ----------
  """
  netcdf2d=NetCDF2D(Input)
  
  # Test metadata info
  info = netcdf2d.info()
  assert info['metadata']['title']==Input['nca']['metadata']['title']

  # Write and read "time" variable
  timeshape = netcdf2d.groups["time"].shape
  timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
  netcdf2d["time","time"] = timevalue
  np.testing.assert_array_equal(netcdf2d["time","time"], timevalue)
  
  # Write and read "bed" variable
  bedshape = netcdf2d.groups["nodes"].shape
  bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
  netcdf2d["nodes","bed"] = bedvalue
  np.testing.assert_array_equal(netcdf2d["nodes","bed"], bedvalue)
  
  # Write and read "a" variable
  sashape = netcdf2d.groups["s"].shape
  savalue = np.arange(np.prod(sashape)).reshape(sashape)
  netcdf2d["s","a"] = savalue
  np.testing.assert_array_equal(netcdf2d["s","a"], savalue)
  
  # Single int and float will be copied based on the index shape
  netcdf2d["s","a",0,100:200] = 0.0
  np.testing.assert_array_equal(netcdf2d["s","a",0,100:200], np.zeros(100))
  
  tashape = netcdf2d.groups["t"].shape
  tavalue = np.arange(np.prod(tashape)).reshape(tashape)
  netcdf2d["t","a"] = tavalue
  np.testing.assert_array_equal(netcdf2d["t","a"], tavalue)
  # netcdf2d.cache.delete()

def test_NetCDF2D_1b():
  # Simple test case to verify upload, clear cache and download from s3
  netcdf2d=NetCDF2D(Input)
  netcdf2d.cache.uploadNCA()
  netcdf2d.cache.uploadNC()
  netcdf2d.cache.clearNCs()
  
  # Write and read "bed" variable
  bedshape = netcdf2d.groups["nodes"].shape
  bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
  netcdf2d["nodes","bed"] = bedvalue
  np.testing.assert_array_equal(netcdf2d["nodes","bed"], bedvalue)
  

if __name__ == "__main__":
  test_NetCDF2D_1()
  # test_NetCDF2D_1b()