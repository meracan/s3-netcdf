import shutil
import pytest
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta




Input = dict(
  name="input2",
  cacheLocation=r"../s3",
  localOnly=True,
  autoUpload=True,
  bucket="merac-dev",
  cacheSize=10.0,
  ncSize=10.0,
  
  nca = dict(
    metadata=dict(title="Input2"),
    dimensions = [
      dict(name="npe" ,value=3),
      dict(name="nelem" ,value=262145),
      dict(name="nnode" ,value=262145),
      dict(name="ntime" ,value=2),
      dict(name="nspectra" ,value=300),
      dict(name="nfreq" ,value=33),
      dict(name="ndir" ,value=36),
    ],
    groups=[
      dict(name="elem",dimensions=["nelem","npe"],variables=[
        dict(name="elem",type="i4", units="" ,standard_name="" ,long_name=""),
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
      dict(name="ss" ,dimensions=["ntime", "nspectra", "nfreq", "ndir"] ,variables=[
        dict(name="e" ,type="f4" ,units="watts" ,standard_name="" ,long_name=""),
        ]),
      dict(name="st" ,dimensions=["nspectra" ,"ntime", "nfreq", "ndir"] ,variables=[
        dict(name="e" ,type="f4" ,units="watts" ,standard_name="" ,long_name=""),
        ]),
    ]
  )
)

def test_NetCDF2D_2():
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
  
  eshape = netcdf2d.getVShape("ss","e")
  evalue = np.arange(np.prod(eshape)).reshape(eshape)
  netcdf2d["ss","e"] = evalue
  np.testing.assert_array_equal(netcdf2d["ss","e"], np.squeeze(evalue))
  np.testing.assert_array_equal(netcdf2d["ss","e",0], np.squeeze(evalue[0]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1], np.squeeze(evalue[1]))
  np.testing.assert_array_equal(netcdf2d["ss","e",:2], np.squeeze(evalue[:2]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1,0], np.squeeze(evalue[1,0]))
  np.testing.assert_array_equal(netcdf2d["ss","e",1,100:200], np.squeeze(evalue[1,100:200]))
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

if __name__ == "__main__":
  test_NetCDF2D_2()