import shutil
import pytest
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta

Input = dict(
  name="input1",
  cacheLocation=r"../s3",
  localOnly=True,
  bucket="uvic-bcwave",
  cacheSize=0.1, # 100kb
  ncSize=1.0, #
  nca = dict(
    metadata=dict(title="Input1"),
    dimensions = {"nnode":1000000,"ntime":2},
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
    Part 1
    localOnly:true, cache files locally and ignore s3 auto commands
    1. create master file and check basic parameters (metadata)
    2. write to partition files
    2b.read and check values from partition files
    3. upload cache files to s3
    3b.delete cache files (check non-existing files)
    4. read (autodownloads from s3 and check values)
    5. delete cache
    5b.delete s3 (check non-existing files in s3)
    
    Part 2
    6. localOnly:false, cache files and upload to s3 automatically
    7. write to partition files
    8. check auto-delete of cache files (exceeding cacheSize)
    9. delete cache (check non-existing files)
    9b.delete s3 (check non-existing files in s3)
    
  ----------
  """
  # 1. Create Master file and check metadata
  netcdf2d=NetCDF2D(Input)
  info = netcdf2d.info()
  assert info['metadata']['title']==Input['nca']['metadata']['title']

  # 2. Write to partition files
  timeshape = netcdf2d.groups["time"].shape
  # timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
  timevalue=(np.datetime64(datetime(2001,3,1))+np.arange(np.prod(timeshape))*np.timedelta64(1, 'h')).astype("datetime64[s]")
  
  netcdf2d["time","time"] = timevalue
  np.testing.assert_array_equal(netcdf2d["time","time"], timevalue)
  
  bedshape = netcdf2d.groups["nodes"].shape
  bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
  netcdf2d["nodes","bed"] = bedvalue
  np.testing.assert_array_equal(netcdf2d["nodes","bed"], bedvalue)
  
  sashape = netcdf2d.groups["s"].shape
  savalue = np.arange(np.prod(sashape)).reshape(sashape)
  netcdf2d["s","a"] = savalue
  np.testing.assert_array_equal(netcdf2d["s","a"], savalue)
  
  netcdf2d["s","a",0,100:200] = 0.0
  np.testing.assert_array_equal(netcdf2d["s","a",0,100:200], np.zeros(100))
  
  # 3. Upload and delete cache files
  netcdf2d.cache.uploadNCA()
  netcdf2d.cache.uploadNC()
  netcdf2d.cache.delete()
  
  netcdf2d.setlocalOnly(False)
  # 4. Download automatically from s3 and check values
  np.testing.assert_array_equal(netcdf2d["s","a",0,100:200], np.zeros(100))
  
  # 5. Delete all
  netcdf2d.cache.delete()
  netcdf2d.s3.delete()
  
  # 6. Part2 cache files and upload to s3 automatically
  Input['localOnly']=False
  netcdf2d=NetCDF2D(Input)
  
  # 7. Write to partition files and upload to s3
  sashape = netcdf2d.groups["s"].shape
  savalue = np.arange(np.prod(sashape)).reshape(sashape)
  netcdf2d["s","a"] = savalue
  np.testing.assert_array_equal(netcdf2d["s","a"], savalue)
  
  # 8. Check auto-delete of cache files (exceeding cacheSize)
  # Should delete 3 files
  assert len(netcdf2d.cache.getNCs())==5
  
  # 9. Delete all
  netcdf2d.cache.delete()
  netcdf2d.s3.delete()
  

if __name__ == "__main__":
  test_NetCDF2D_1()