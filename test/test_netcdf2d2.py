import pytest
import sys
import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime, timedelta

 
Input = dict(
  name="s3input2",
  cacheLocation=r"../s3",
  localOnly=True,
  
  bucket="uvic-bcwave",
  cacheSize=10.0,
  ncSize=1.0,
  squeeze=True,
  
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
      nfeature=3,
      nchar=32,
      nD=11,
      nV=5,
    ),
    groups=dict(
      elem=dict(dimensions=["nelem","npe"],variables=dict(
        elem=dict(type="i4", units="" ,standard_name="" ,long_name=""),
      )),
      time=dict(dimensions=["ntime"],variables=dict(
        time=dict(type="f8",units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="" ,long_name=""),
        )),
      node=dict(dimensions=["nnode"],variables=dict(
        bed=dict(type="f4",units="m" ,standard_name="" ,long_name=""),
        friction=dict(type="f4" ,units="" ,standard_name="" ,long_name=""),
        )),
      feature=dict(dimensions=["nfeature",'nchar'],variables=dict(
        name=dict(type="S2",units="" ,standard_name="Feature Name" ,long_name=""),
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
      sV=dict(dimensions=["nD" ,"nV", "nnode"] ,variables=dict(
        f=dict(type="f4" ,units="watts" ,standard_name="" ,long_name=""),
      )),
      u1s=dict(dimensions=["ntime", "nnode"] ,variables=dict(
        f=dict(type="u1",ftype="f4",max=25.5,min=0.0,units="m" ,standard_name="" ,long_name=""),
        g=dict(type="u1",max=25.5,min=0.0,units="m" ,standard_name="" ,long_name=""),
        h=dict(type="u4",max=25.5,min=0.0,units="m" ,standard_name="" ,long_name=""),
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
  # timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
  timevalue = np.datetime64(datetime(2001,3,1))+np.arange(np.prod(timeshape))*np.timedelta64(1, 'h')
  netcdf2d["time","time"] = timevalue.astype("datetime64[s]")
  np.testing.assert_array_equal(netcdf2d["time","time"], timevalue)
  
  bedshape = netcdf2d.groups["node"].shape
  bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
  netcdf2d["node","bed"] = bedvalue
  np.testing.assert_array_equal(netcdf2d["node","bed"], bedvalue)
  
  netcdf2d['feature','name']=np.array(['USA', 'Japan', 'UK'])
  np.testing.assert_array_equal(netcdf2d["feature","name"], ['USA', 'Japan', 'UK'])
  
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
  
  
  netcdf2d["sV","f",0] = 0.0
  netcdf2d["sV","f",6:11] = np.ones((5,5,262145))
  
  
  # Testing transform
  netcdf2d["u1s","f"] = savalue
  np.testing.assert_array_equal(netcdf2d["u1s","f"], np.clip(savalue,0,25.5))
  netcdf2d["u1s","g"] = savalue
  np.testing.assert_array_equal(netcdf2d["u1s","g"],  np.clip(savalue.astype('f4'),0,25.5))
  
  value=savalue/(np.prod(sashape))*25.5
  netcdf2d["u1s","h"] = value
  np.testing.assert_almost_equal(netcdf2d["u1s","h"],value,decimal=8)
  
  attributes=netcdf2d.variables['h']
  f_u=NetCDF2D.transform(attributes,value,set=True)
  f_f=NetCDF2D.transform(attributes,f_u,set=False)
  np.testing.assert_almost_equal(value,f_f,decimal=8)
  
  assert netcdf2d.getGroupsByVariable('a')==['s','t'] 
  
  netcdf2d.cache.delete()

def test_NetCDF2D_2_query():
  netcdf2d=NetCDF2D(Input)
  elemshape = netcdf2d.groups["elem"].shape
  
  elemvalue = np.arange(np.prod(elemshape)).reshape(elemshape)
  
  netcdf2d["elem","elem"] = elemvalue
  np.testing.assert_array_equal(netcdf2d.query({"group":"elem","variable":"elem"}), elemvalue)
  np.testing.assert_array_equal(netcdf2d.query({"variable":"elem"}), elemvalue)
  
  sashape = netcdf2d.groups["s"].shape
  savalue = np.arange(np.prod(sashape)).reshape(sashape)
  netcdf2d["s","a"] = savalue
  netcdf2d["t","a"] = savalue.T
  
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a"}), savalue)
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":0}), savalue[0])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":0,"inode":":"}), savalue[0,:])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":":","inode":0}), savalue[:,0])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":0,"inode":0}), savalue[0,0])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":0,"inode":131073}), savalue[0,131073])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":0,"inode":262144}), savalue[0,262144])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":1,"inode":100}), savalue[1,100])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":1,"inode":131073}), savalue[1,131073])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":1,"inode":262144}), savalue[1,262144])
  
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":"0:2","inode":0}), savalue[0:2,0])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":"0:2","inode":131073}), savalue[0:2,131073])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":"0:2","inode":262144}), savalue[0:2,262144])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":"[0,1]","inode":100}), savalue[[0,1],100])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":"[0,1]","inode":131073}), savalue[[0,1],131073])
  np.testing.assert_array_equal(netcdf2d.query({"group":"s","variable":"a","itime":"[0,1]","inode":262144}), savalue[[0,1],262144])
  
  
  # np.testing.assert_array_equal(netcdf2d.query({"variable":"a"}), savalue.T) # .T its gets the "t" group since it's the latter one
  np.testing.assert_array_equal(netcdf2d.query({"variable":"a","itime":0}), savalue[0])
  np.testing.assert_array_equal(netcdf2d.query({"variable":"a","itime":"0:2","inode":0}), savalue[0:2,0])
  np.testing.assert_array_equal(netcdf2d.query({"variable":"a","inode":0}), savalue.T[0])
  
  netcdf2d.cache.delete()
  
  

  

if __name__ == "__main__":
  test_NetCDF2D_2()
  test_NetCDF2D_2_query()
