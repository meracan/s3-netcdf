import pytest
import numpy as np
from s3netcdf import S3NetCDF
from datetime import datetime, timedelta

Input = {
  "name":"input4",
  "cacheLocation":r"../s3",
  "localOnly":True,
  "bucket":"uvic-bcwave",
  "cacheSize":10.0,
  "ncSize":10.0,
  "squeeze":True,
  "nca":{
    "metadata":{"title":"Input4"},
    "dimensions":{"npe":3,"nelem":262145,"nnode":262145,"ntime":110,"nspectra":300,"nfreq":33,"ndir":36,"nstation":100,"nsnode":20},
    "groups":{
      "elem":{"dimensions":["nelem","npe"],"variables":{
        "elem":{"type":"i4", "units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "time":{"dimensions":["ntime"],"variables":{
        "time":{"type":"M","standard_name":"" ,"long_name":""},
      }},
      "node":{"dimensions":["nnode"],"variables":{
        "bed":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
        "friction":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "s":{"dimensions":["ntime","nnode"],"variables":{
        "a":{"type":"f8","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "spc":{"dimensions":["nstation", "nsnode","ntime","nfreq","ndir"],"variables":{
        "energy":{"type":"f8","units":"" ,"standard_name":"" ,"long_name":""},
      }},
    }
  }
}


def test_NetCDF2D_4():
  """
  Advanced testing to test index assignment
  ----------
  """
  with S3NetCDF(Input) as s3netcdf:
    elemshape = s3netcdf.groups["s"].shape
    
    value = np.arange(np.prod(elemshape)).reshape(elemshape)
    
    
    s3netcdf["s","a",0] = value[0]
    np.testing.assert_array_equal(s3netcdf["s","a",0], value[0])
    
    s3netcdf["s","a",50] = value[50]
    np.testing.assert_array_equal(s3netcdf["s","a",50], value[50])
    
    s3netcdf["s","a",10:40] = value[10:40]
    np.testing.assert_array_equal(s3netcdf["s","a",10:40], value[10:40])
    
    s3netcdf["s","a",100:101] = value[100:101]
    np.testing.assert_array_equal(s3netcdf["s","a",100:101],np.squeeze(value[100:101]))  
  
    s3netcdf["s","a"] = value
    np.testing.assert_array_equal(s3netcdf["s","a"], value)
    
    value2 = np.arange(20*110*33*36).reshape((1,20,110,33,36))
    
    s3netcdf["spc","energy",0,:] = value2
    np.testing.assert_array_equal(s3netcdf["spc","energy",0,0:20], np.squeeze(value2))
    np.testing.assert_array_equal(s3netcdf["spc","energy",0], np.squeeze(value2))
    
    shape=s3netcdf.groups['s'].shape
    child=s3netcdf.groups['s'].child
    n=shape[0]
    step=child[0]
    
    
    for i in range(0,n,step):
      j=np.minimum(n,i+step)
      s3netcdf["s","a",i:j] = np.ones(child)
    np.testing.assert_array_equal(s3netcdf["s","a",0], np.ones((262145)))
    
    
    
    s3netcdf.cache.delete()


if __name__ == "__main__":
  test_NetCDF2D_4()