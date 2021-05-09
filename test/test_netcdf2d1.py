import shutil
import os
import pytest
import numpy as np
from s3netcdf import S3NetCDF
from datetime import datetime, timedelta
import time
Input={
  "name":"s3input1",
  "cacheLocation":r"../s3",
  "localOnly":True,
  "bucket":"uvic-bcwave",
  "cacheSize":0.1,
  "ncSize":1,
  "nca":{
     "metadata":{"title":"Input1"},
     "dimensions": {"nnode":1000000,"ntime":2},
     "groups":{
      "time":{"dimensions":["ntime"],"variables":{
          "time":{"type":"M", "standard_name":"Datetime" ,"long_name":"Datetime"}
        }
      },
      "node":{"dimensions":["nnode"],"variables":{
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
  }
}


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
  with S3NetCDF(Input,"r") as s3netcdf:
    assert s3netcdf.obj['metadata']['title']==Input['nca']['metadata']['title']
    
    # 2. Write to partition files
    timeshape = s3netcdf["time"].shape
    # timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
    timevalue=(np.datetime64(datetime(2001,3,1))+np.arange(np.prod(timeshape))*np.timedelta64(1, 'h')).astype("datetime64[s]")
    s3netcdf["time","time"] = timevalue
    np.testing.assert_array_equal(s3netcdf["time","time"], timevalue)
    
    bedshape = s3netcdf["node"].shape
    bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
    s3netcdf["node","bed"] = bedvalue
    np.testing.assert_array_equal(s3netcdf["node","bed"], bedvalue)
    
    sashape = s3netcdf["s"].shape
    savalue = np.arange(np.prod(sashape)).reshape(sashape)
    s3netcdf["s","a"] = savalue
    np.testing.assert_array_equal(s3netcdf["s","a"], savalue)
    
    s3netcdf["s","a",0,100:200] = 0.0
    np.testing.assert_array_equal(s3netcdf["s","a",0,100:200], np.zeros((1,100)))
    
    # 3. Upload and delete cache files
    s3netcdf.cache.uploadNCA()
    s3netcdf.cache.uploadNC()
    s3netcdf.cache.delete()
    
    s3netcdf.setlocalOnly(False)
    # 4. Download automatically from s3 and check values
    tvalue = np.arange(np.prod(sashape)).reshape(sashape)
    tvalue[0,100:200]= 0.0
    np.testing.assert_array_equal(s3netcdf["s","a"], tvalue)
    
    # 5. Delete all
    s3netcdf.cache.delete()
    s3netcdf.s3.delete()
  
  # 6 Update Metadata
  with S3NetCDF(Input,"r+") as s3netcdf:
    s3netcdf.updateMetadata({"subname":"value"})
  
  # Check updated metadata
  with S3NetCDF(Input,"r") as s3netcdf:
    assert s3netcdf.obj['metadata']['title']==Input['nca']['metadata']['title']
    assert s3netcdf.obj['metadata']['subname']=="value"

def test_NetCDF2D_1b():
  Input['localOnly']=False
  with S3NetCDF(Input,"r") as s3netcdf:
    # 7. Write to partition files and upload to s3
    sashape = s3netcdf.groups["s"].shape
    
    savalue = np.arange(np.prod(sashape)).reshape(sashape)
    s3netcdf["s","a"] = savalue
    np.testing.assert_array_equal(s3netcdf["s","a"], savalue)
    
    # 8. Check auto-delete of cache files (exceeding cacheSize)
    #Should delete 3 files
    assert len(s3netcdf.cache.getNCs())==0
    
    # 9. Delete all
    s3netcdf.cache.delete()
    s3netcdf.s3.delete()
  

if __name__ == "__main__":
  test_NetCDF2D_1()
  test_NetCDF2D_1b()