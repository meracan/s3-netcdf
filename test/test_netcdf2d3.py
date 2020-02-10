import pytest
import os
import json
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta

def test_NetCDF2D_3():
  json_file =  open(os.path.join(os.path.dirname(__file__),'test3.json'))
  Input = json.load(json_file)
  json_file.close()
  
  netcdf2d=NetCDF2D(Input)
  info = netcdf2d.info()
  assert info['metadata']['title']==Input['nca']['metadata']['title']
  
  timeshape = netcdf2d.groups["time"].shape
  timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
  netcdf2d["time","time"] = timevalue
  np.testing.assert_array_equal(netcdf2d["time","time"], timevalue)
  
  json_file =  open(os.path.join(os.path.dirname(__file__),'test3b.json'))
  Input2 = json.load(json_file)
  json_file.close()
  net=NetCDF2D(Input2)
  np.testing.assert_array_equal(net["time","time"], timevalue)
  
  net.cache.delete()
  
  
if __name__ == "__main__":
  test_NetCDF2D_3()