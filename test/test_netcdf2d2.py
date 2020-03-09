import pytest
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta

Input = dict(
  name="input2",
  cacheLocation=r"../s3",
  localOnly=True,
  
  bucket="merac-dev",
  cacheSize=10.0,
  ncSize=1.0,
  
  nca = dict(
    metadata=dict(title="Input2"),
    dimensions = dict(
      npe=3,
      nelem=262145,
      nnode=262145,
      ntime=2,
      nspectra=300,
      nfreq=33,
      ndir=36,
    ),
    groups=dict(
      elem=dict(dimensions=["nelem","npe"],variables=dict(
        elem=dict(type="i4", units="" ,standard_name="" ,long_name=""),
      )),
      time=dict(dimensions=["ntime"],variables=dict(
        time=dict(type="f4",units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="" ,long_name=""),
        )),
      nodes=dict(dimensions=["nnode"],variables=dict(
        bed=dict(type="f4",units="m" ,standard_name="" ,long_name=""),
        friction=dict(type="f4" ,units="" ,standard_name="" ,long_name=""),
        )),
      s=dict(dimensions=["ntime", "nnode"] ,variables=dict(
        a=dict(type="f4",units="m" ,standard_name="" ,long_name=""),
        )),
      t=dict(dimensions=["nnode" ,"ntime"] ,variables=dict(
        a=dict(type="f4",units="m" ,standard_name="" ,long_name=""),
        )),
      ss=dict(dimensions=["ntime", "nspectra", "nfreq", "ndir"] ,variables=dict(
        e=dict(type="f4" ,units="watts" ,standard_name="" ,long_name=""),
        )),
      st=dict(dimensions=["nspectra" ,"ntime", "nfreq", "ndir"] ,variables=dict(
        e=dict(type="f4" ,units="watts" ,standard_name="" ,long_name=""),
        )),
    )
  )
)

def test_NetCDF2D_2():
  """
  Advanced testing to test index assignment
  ----------
  """
  netcdf2d=NetCDF2D(Input)
  elemshape = netcdf2d.groups["elem"].shape
  elemvalue = np.arange(np.prod(elemshape)).reshape(elemshape)
  netcdf2d["elem","elem"] = elemvalue
  np.testing.assert_array_equal(netcdf2d["elem","elem"], elemvalue)
  
  timeshape = netcdf2d.groups["time"].shape
  timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
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
  np.testing.assert_array_equal(netcdf2d["s","a",0], savalue[0])
  np.testing.assert_array_equal(netcdf2d["s","a",0,:], savalue[0,:])
  np.testing.assert_array_equal(netcdf2d["s","a",:,0], savalue[:,0])
  np.testing.assert_array_equal(netcdf2d["s","a",0,0], savalue[0,0])
  np.testing.assert_array_equal(netcdf2d["s","a",0,131073], savalue[0,131073])
  np.testing.assert_array_equal(netcdf2d["s","a",0,262144], savalue[0,262144])
  np.testing.assert_array_equal(netcdf2d["s","a",1,100], savalue[1,100])
  np.testing.assert_array_equal(netcdf2d["s","a",1,131073], savalue[1,131073])
  np.testing.assert_array_equal(netcdf2d["s","a",1,262144], savalue[1,262144])
  
  np.testing.assert_array_equal(netcdf2d["s","a",0:2,0], savalue[0:2,0])
  np.testing.assert_array_equal(netcdf2d["s","a",0:2,131073], savalue[0:2,131073])
  np.testing.assert_array_equal(netcdf2d["s","a",0:2,262144], savalue[0:2,262144])
  np.testing.assert_array_equal(netcdf2d["s","a",[0,1],100], savalue[[0,1],100])
  np.testing.assert_array_equal(netcdf2d["s","a",[0,1],131073], savalue[[0,1],131073])
  np.testing.assert_array_equal(netcdf2d["s","a",[0,1],262144], savalue[[0,1],262144])
  
  np.testing.assert_array_equal(netcdf2d["s","a",0:2,10:20], savalue[0:2,10:20])
  np.testing.assert_array_equal(netcdf2d["s","a",0:2,[0,262144]], savalue[0:2,[0,262144]])
  np.testing.assert_array_equal(netcdf2d["s","a",1:2,131060:262144], np.squeeze(savalue[1:2,131060:262144])) # For some reason, numpy does not squeeze this
  np.testing.assert_array_equal(netcdf2d["s","a",1,131060:262144], savalue[1,131060:262144])
  np.testing.assert_array_equal(netcdf2d["s","a",0:2,[0,131060,131076,262144]], savalue[0:2,[0,131060,131076,262144]])
  
  z10=np.zeros(10)
  netcdf2d["s","a",0,0:10] = z10
  netcdf2d["s","a",0,30:50] = 6.0
  netcdf2d["s","a",0,131073:131083] = z10
  netcdf2d["s","a",0:2,100:110] = np.repeat(z10,2)
  np.testing.assert_array_equal(netcdf2d["s","a",0,0:10], z10)
  np.testing.assert_array_equal(netcdf2d["s","a",0,30:50], np.zeros(20)+6)
  np.testing.assert_array_equal(netcdf2d["s","a",0,131073:131083], z10)
  np.testing.assert_array_equal(netcdf2d["s","a",0,100:110], z10)
  np.testing.assert_array_equal(netcdf2d["s","a",1,100:110], z10)
  
  eshape = netcdf2d.groups["ss"].shape
  evalue = np.arange(np.prod(eshape)).reshape(eshape)
  netcdf2d["ss","e"] = evalue
  np.testing.assert_array_equal(netcdf2d["ss","e"], np.squeeze(evalue))
  np.testing.assert_array_equal(netcdf2d["ss","e",0], np.squeeze(evalue[0]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1], np.squeeze(evalue[1]))
  np.testing.assert_array_equal(netcdf2d["ss","e",:2], np.squeeze(evalue[:2]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1,0], np.squeeze(evalue[1,0]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1,100:200], np.squeeze(evalue[1,100:200]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1:2,100:200], np.squeeze(evalue[1:2,100:200]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1,[1,101,201]], np.squeeze(evalue[1,[1,101,201]]))
  np.testing.assert_array_equal(netcdf2d["ss","e",:2,299], np.squeeze(evalue[:2,299]))
  np.testing.assert_array_equal(netcdf2d["ss","e",:2,100:200], np.squeeze(evalue[:2,100:200]))
  np.testing.assert_array_equal(netcdf2d["ss","e",:2,[1,101,201]], np.squeeze(evalue[:2,[1,101,201]]))
  np.testing.assert_array_equal(netcdf2d["ss","e",0,[0,100,200],0:10], np.squeeze(evalue[0,[0,100,200],0:10]))
  np.testing.assert_array_equal(netcdf2d["ss","e",0,0,0,0], np.squeeze(evalue[0,0,0,0]))
  np.testing.assert_array_equal(netcdf2d["ss","e",0,200], np.squeeze(evalue[0,200]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1,200], np.squeeze(evalue[1,200]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1,200,20:30,10:15], np.squeeze(evalue[1,200,20:30,10:15]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1,[100,250],20:30,10:15], np.squeeze(evalue[1,[100,250],20:30,10:15]))
  
  
  netcdf2d["ss","e",0,0:10,0,0] = z10
  netcdf2d["ss","e",1,200:210,0,0] = z10
  netcdf2d["ss","e",0:2,250:260,0,0] = np.zeros(20).reshape((2,10))
  np.testing.assert_array_equal(netcdf2d["ss","e",0,0:10,0,0], z10)
  np.testing.assert_array_equal(netcdf2d["ss","e",1,200:210,0,0], z10)
  np.testing.assert_array_equal(netcdf2d["ss","e",0:2,250:260,0,0], np.zeros(20).reshape((2,10)))
  netcdf2d["ss","e",0,0,0] = np.arange(36)
  np.testing.assert_array_equal(netcdf2d["ss","e",0,0,0], np.arange(36))
  
  netcdf2d.cache.delete()

if __name__ == "__main__":
  test_NetCDF2D_2()