import pytest
import os
import json
import numpy as np
from s3netcdf import S3NetCDF
from datetime import datetime, timedelta

def test_NetCDF2D_3():
  json_file =  open(os.path.join(os.path.dirname(__file__),'test3.json'))
  Input = json.load(json_file)
  json_file.close()
  
  s3netcdf=S3NetCDF(Input)
  assert s3netcdf.obj['metadata']['title']==Input['nca']['metadata']['title']

  
  timeshape = s3netcdf.groups["time"].shape
  timevalue=np.datetime64(datetime(2001,3,1))+np.arange(np.prod(timeshape))*np.timedelta64(1, 'h')
  s3netcdf["time","time"] = timevalue.astype("datetime64[s]")
  np.testing.assert_array_equal(s3netcdf["time","time"], timevalue)
  
  json_file =  open(os.path.join(os.path.dirname(__file__),'test3b.json'))
  Input2 = json.load(json_file)
  json_file.close()
  net=S3NetCDF(Input2)
  np.testing.assert_array_equal(net["time","time"], timevalue)
  
  net.cache.delete()
  
  
if __name__ == "__main__":
  test_NetCDF2D_3()