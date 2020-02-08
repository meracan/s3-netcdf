import shutil
import pytest
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta
import time



Input = dict(
  name="input3",
  cacheLocation=r"../s3",
  localOnly=False,
  autoUpload=True,
  bucket="merac-dev",
  cacheSize=2.0,
  ncSize=100.0,
  metadata=dict(
    title="Input3"
  ),
  nca = dict(
    dimensions = [
      dict(name="npe" ,value=3),
      dict(name="nelem" ,value=3000000),
      dict(name="nnode" ,value=3000000),
      dict(name="ntime" ,value=6*24*45),
    ],
    groups=[
      dict(name="elem",dimensions=["nelem","npe"],variables=[
        dict(name="elem",type="i4", units="" ,standard_name="" ,long_name=""),
        ]),
      dict(name="time",dimensions=["ntime"],variables=[
        dict(name="time",type="f8",units="hours since 2000-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="" ,long_name=""),
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

def perf_NetCDF2D_3():
  shutil.rmtree(Input['cacheLocation'])
  netcdf2d=NetCDF2D(Input)
  elemshape = netcdf2d.getVShape("elem","elem")
  elemvalue = np.arange(np.prod(elemshape)).reshape(elemshape)
  netcdf2d["elem","elem"] = elemvalue
  np.testing.assert_array_equal(netcdf2d["elem","elem"], elemvalue)
  
  timeshape = netcdf2d.getVShape("time","time")
  timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
  netcdf2d["time","time"] = timevalue
  np.testing.assert_array_equal(netcdf2d["time","time"], timevalue)
  
  bedshape = netcdf2d.getVShape("nodes","bed")
  bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
  netcdf2d["nodes","bed"] = bedvalue
  np.testing.assert_array_equal(netcdf2d["nodes","bed"], bedvalue)
  
  sashape = netcdf2d.getVShape("s","a")
  start = 0
  end = 1
  step=10
  for i in range(0,sashape[0],step):
    print("{} of {} - {}hrs".format(i,sashape[0],(end-start)*(sashape[0]-1-i)/step/60./60.))
    start = time.time()
    netcdf2d["s","a",i:i+step] = np.arange(sashape[1]*step)+i*step
    end = time.time()


def perf_NetCDF2D_3b():
  netcdf2d=NetCDF2D(Input)
  sashape = netcdf2d.getVShape("s","a")
  np.testing.assert_array_equal(netcdf2d["s","a",0], np.arange(sashape[1])+0)
  np.testing.assert_array_equal(netcdf2d["s","a",10], np.arange(sashape[1])+10*10)
  np.testing.assert_array_equal(netcdf2d["s","a",20], np.arange(sashape[1])+20*10)
  np.testing.assert_array_equal(netcdf2d["s","a",30], np.arange(sashape[1])+30*10)
  np.testing.assert_array_equal(netcdf2d["s","a",40], np.arange(sashape[1])+40*10)
  np.testing.assert_array_equal(netcdf2d["s","a",50], np.arange(sashape[1])+50*10)
  
  start = time.time()
  netcdf2d["s","a",0]
  end = time.time()
  print(end-start)
  
if __name__ == "__main__":
  # perf_NetCDF2D_3()
  perf_NetCDF2D_3b()