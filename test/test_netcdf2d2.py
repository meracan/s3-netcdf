import pytest
import os
import shutil
import sys
import numpy as np
from s3netcdf import S3NetCDF
from datetime import datetime, timedelta

Input={
  "name":"s3input2",
  "cacheLocation":r"../s3",
  "localOnly":True,
  "bucket":"uvic-bcwave",
  "cacheSize":10.0,
  "ncSize":1.0,
  "squeeze":True,
  "nca":{
    "metadata":{
      "title":"Input2",
      "containers":{"mesh":["elem","node"],"hour":["s","t"],"spec":["ss","st"]},
    },
    "dimensions":{"npe":3,"nelem":262145,"nnode":262145,"ntime":2,"nspectra":300,"nfreq":33,"ndir":36,"nfeature":3,"nchar":32,"nD":11,"nV":5,"ndnode":100,"ndtime":100},
    "groups":{
      "elem":{"dimensions":["nelem","npe"],"variables":{
        "elem":{"type":"i4", "units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "time":{"dimensions":["ntime"],"variables":{
        "time":{"type":"f8","ftype":"M","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "node":{"dimensions":["nnode"],"variables":{
        "bed":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
        "friction":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "feature":{"dimensions":["nfeature","nchar"],"variables":{
        "name":{"type":"S1","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "s":{"dimensions":["ntime","nnode"],"variables":{
        "a":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "t":{"dimensions":["nnode","ntime"],"variables":{
        "a":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "dt":{"dimensions":["ndnode","ndtime"],"variables":{
        "a":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "ss":{"dimensions":["ntime", "nspectra", "nfreq", "ndir"],"variables":{
        "e":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "st":{"dimensions":["nspectra" ,"ntime", "nfreq", "ndir"],"variables":{
        "e":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "sV":{"dimensions":["nD" ,"nV", "nnode"],"variables":{
        "f":{"type":"f4","units":"" ,"standard_name":"" ,"long_name":""},
      }},
      "u1s":{"dimensions":["ntime", "nnode"],"variables":{
        "f":{"type":"f4","stype":"u1","max":25.5,"min":0.0,"units":"" ,"standard_name":"" ,"long_name":""},
        "g":{"type":"f4","stype":"u2","max":25.5,"min":0.0,"units":"" ,"standard_name":"" ,"long_name":""},
        "h":{"type":"f8","stype":"u4","max":25.5,"min":0.0,"units":"" ,"standard_name":"" ,"long_name":""},
      }},       
    }
  }
}


def test_NetCDF2D_2():
  """
  Advanced testing to test index assignment
  ----------
  """
  
  # shutil.rmtree(os.path.join('../s3','s3input2'))
  with S3NetCDF(Input) as s3netcdf:
    elemshape = s3netcdf.groups["elem"].shape
    elemvalue = np.arange(np.prod(elemshape)).reshape(elemshape)
    
    s3netcdf["elem","elem"] = elemvalue
    np.testing.assert_array_equal(s3netcdf["elem","elem"], elemvalue)
    
    timeshape = s3netcdf.groups["time"].shape
    # timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
    timevalue = np.datetime64(datetime(2001,3,1))+np.arange(np.prod(timeshape))*np.timedelta64(1, 'h')
    s3netcdf["time","time"] = timevalue.astype("datetime64[s]")
    np.testing.assert_array_equal(s3netcdf["time","time"], timevalue)
    
    bedshape = s3netcdf.groups["node"].shape
    bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
    s3netcdf["node","bed"] = bedvalue
    np.testing.assert_array_equal(s3netcdf["node","bed"], bedvalue)
    
    s3netcdf['feature','name']=np.array(['USA', 'Japan', 'UK'])
    np.testing.assert_array_equal(s3netcdf["feature","name"], ['USA', 'Japan', 'UK'])
    
    sashape = s3netcdf.groups["s"].shape
    savalue = np.arange(np.prod(sashape)).reshape(sashape)
    s3netcdf["s","a"] = savalue
    np.testing.assert_array_equal(s3netcdf["s","a"], savalue)
    np.testing.assert_array_equal(s3netcdf["s","a",0], savalue[0])
    np.testing.assert_array_equal(s3netcdf["s","a",0,:], savalue[0,:])
    np.testing.assert_array_equal(s3netcdf["s","a",:,0], savalue[:,0])
    np.testing.assert_array_equal(s3netcdf["s","a",0,0], savalue[0,0])
    np.testing.assert_array_equal(s3netcdf["s","a",0,131073], savalue[0,131073])
    np.testing.assert_array_equal(s3netcdf["s","a",0,262144], savalue[0,262144])
    np.testing.assert_array_equal(s3netcdf["s","a",1,100], savalue[1,100])
    np.testing.assert_array_equal(s3netcdf["s","a",1,131073], savalue[1,131073])
    np.testing.assert_array_equal(s3netcdf["s","a",1,262144], savalue[1,262144])
    
    np.testing.assert_array_equal(s3netcdf["s","a",0:2,0], savalue[0:2,0])
    np.testing.assert_array_equal(s3netcdf["s","a",0:2,131073], savalue[0:2,131073])
    np.testing.assert_array_equal(s3netcdf["s","a",0:2,262144], savalue[0:2,262144])
    np.testing.assert_array_equal(s3netcdf["s","a",[0,1],100], savalue[[0,1],100])
    np.testing.assert_array_equal(s3netcdf["s","a",[0,1],131073], savalue[[0,1],131073])
    np.testing.assert_array_equal(s3netcdf["s","a",[0,1],262144], savalue[[0,1],262144])
    
    np.testing.assert_array_equal(s3netcdf["s","a",0:2,10:20], savalue[0:2,10:20])
    np.testing.assert_array_equal(s3netcdf["s","a",0:2,[0,262144]], savalue[0:2,[0,262144]])
    np.testing.assert_array_equal(s3netcdf["s","a",1:2,131060:262144], np.squeeze(savalue[1:2,131060:262144])) # For some reason, numpy does not squeeze this
    np.testing.assert_array_equal(s3netcdf["s","a",1,131060:262144], savalue[1,131060:262144])
    np.testing.assert_array_equal(s3netcdf["s","a",0:2,[0,131060,131076,262144]], savalue[0:2,[0,131060,131076,262144]])
    
    z10=np.zeros(10)
    s3netcdf["s","a",0,0:10] = z10
    s3netcdf["s","a",0,30:50] = 6.0
    s3netcdf["s","a",0,131073:131083] = z10
    s3netcdf["s","a",0:2,100:110] = np.repeat(z10,2)
    np.testing.assert_array_equal(s3netcdf["s","a",0,0:10], z10)
    np.testing.assert_array_equal(s3netcdf["s","a",0,30:50], np.zeros(20)+6)
    np.testing.assert_array_equal(s3netcdf["s","a",0,131073:131083], z10)
    np.testing.assert_array_equal(s3netcdf["s","a",0,100:110], z10)
    np.testing.assert_array_equal(s3netcdf["s","a",1,100:110], z10)
    
    
    eshape = s3netcdf.groups["ss"].shape
    evalue = np.arange(np.prod(eshape)).reshape(eshape)
    s3netcdf["ss","e"] = evalue
    np.testing.assert_array_equal(s3netcdf["ss","e"], np.squeeze(evalue))
    np.testing.assert_array_equal(s3netcdf["ss","e",0], np.squeeze(evalue[0]))
    np.testing.assert_array_equal(s3netcdf["ss","e",1], np.squeeze(evalue[1]))
    np.testing.assert_array_equal(s3netcdf["ss","e",:2], np.squeeze(evalue[:2]))
    np.testing.assert_array_equal(s3netcdf["ss","e",1,0], np.squeeze(evalue[1,0]))
    np.testing.assert_array_equal(s3netcdf["ss","e",1,100:200], np.squeeze(evalue[1,100:200]))
    np.testing.assert_array_equal(s3netcdf["ss","e",1:2,100:200], np.squeeze(evalue[1:2,100:200]))
    np.testing.assert_array_equal(s3netcdf["ss","e",1,[1,101,201]], np.squeeze(evalue[1,[1,101,201]]))
    np.testing.assert_array_equal(s3netcdf["ss","e",:2,299], np.squeeze(evalue[:2,299]))
    np.testing.assert_array_equal(s3netcdf["ss","e",:2,100:200], np.squeeze(evalue[:2,100:200]))
    np.testing.assert_array_equal(s3netcdf["ss","e",:2,[1,101,201]], np.squeeze(evalue[:2,[1,101,201]]))
    np.testing.assert_array_equal(s3netcdf["ss","e",0,[0,100,200],0:10], np.squeeze(evalue[0,[0,100,200],0:10]))
    np.testing.assert_array_equal(s3netcdf["ss","e",0,0,0,0], np.squeeze(evalue[0,0,0,0]))
    np.testing.assert_array_equal(s3netcdf["ss","e",0,200], np.squeeze(evalue[0,200]))
    np.testing.assert_array_equal(s3netcdf["ss","e",1,200], np.squeeze(evalue[1,200]))
    np.testing.assert_array_equal(s3netcdf["ss","e",1,200,20:30,10:15], np.squeeze(evalue[1,200,20:30,10:15]))
    np.testing.assert_array_equal(s3netcdf["ss","e",1,[100,250],20:30,10:15], np.squeeze(evalue[1,[100,250],20:30,10:15]))
    
    
    s3netcdf["ss","e",0,0:10,0,0] = z10
    s3netcdf["ss","e",1,200:210,0,0] = z10
    s3netcdf["ss","e",0:2,250:260,0,0] = np.zeros(20).reshape((2,10))
    np.testing.assert_array_equal(s3netcdf["ss","e",0,0:10,0,0], z10)
    np.testing.assert_array_equal(s3netcdf["ss","e",1,200:210,0,0], z10)
    np.testing.assert_array_equal(s3netcdf["ss","e",0:2,250:260,0,0], np.zeros(20).reshape((2,10)))
    s3netcdf["ss","e",0,0,0] = np.arange(36)
    np.testing.assert_array_equal(s3netcdf["ss","e",0,0,0], np.arange(36))
    
    
    s3netcdf["sV","f",0] = 0.0
    s3netcdf["sV","f",6:11] = np.ones((5,5,262145))
    
    
    # Testing transform
    s3netcdf["u1s","f"] = savalue
    np.testing.assert_almost_equal(s3netcdf["u1s","f"], np.clip(savalue.astype('f4'),0,25.5),decimal=8)
    s3netcdf["u1s","g"] = savalue
    np.testing.assert_almost_equal(s3netcdf["u1s","g"],  np.clip(savalue.astype('f4'),0,25.5),decimal=8)
    
    value=savalue/(np.prod(sashape))*25.5
    s3netcdf["u1s","h"] = value
    np.testing.assert_almost_equal(s3netcdf["u1s","h"],value,decimal=8)
    
    
    np.testing.assert_array_equal(np.sort(s3netcdf.getGroupsByVariable('a')),['dt','s','t'])
    
    s3netcdf.cache.delete()

def test_NetCDF2D_2_query():
  with S3NetCDF(Input) as s3netcdf:
    elemshape = s3netcdf.groups["elem"].shape
    elemvalue = np.arange(np.prod(elemshape)).reshape(elemshape)
    
    s3netcdf["elem","elem"] = elemvalue
    np.testing.assert_array_equal(s3netcdf.query({"group":"elem","variable":"elem"}), elemvalue)
    np.testing.assert_array_equal(s3netcdf.query({"variable":"elem"}), elemvalue)
    
    sashape = s3netcdf.groups["s"].shape
    savalue = np.arange(np.prod(sashape)).reshape(sashape)
    s3netcdf["s","a"] = savalue
    s3netcdf["t","a"] = savalue.T
    
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a"}), savalue)
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":0}), savalue[0])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":0,"inode":":"}), savalue[0,:])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":":","inode":0}), savalue[:,0])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":0,"inode":0}), savalue[0,0])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":0,"inode":131073}), savalue[0,131073])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":0,"inode":262144}), savalue[0,262144])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":1,"inode":100}), savalue[1,100])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":1,"inode":131073}), savalue[1,131073])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":1,"inode":262144}), savalue[1,262144])
    
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":"0:2","inode":0}), savalue[0:2,0])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":"0:2","inode":131073}), savalue[0:2,131073])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":"0:2","inode":262144}), savalue[0:2,262144])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":"[0,1]","inode":100}), savalue[[0,1],100])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":"[0,1]","inode":131073}), savalue[[0,1],131073])
    np.testing.assert_array_equal(s3netcdf.query({"group":"s","variable":"a","itime":"[0,1]","inode":262144}), savalue[[0,1],262144])
    
    
    # # np.testing.assert_array_equal(s3netcdf.query({"variable":"a"}), savalue.T) # .T its gets the "t" group since it's the latter one
    np.testing.assert_array_equal(s3netcdf.query({"variable":"a","itime":0}), savalue[0])
    np.testing.assert_array_equal(s3netcdf.query({"variable":"a","itime":"0:2","inode":0}), savalue[0:2,0])
    np.testing.assert_array_equal(s3netcdf.query({"variable":"a","inode":0}), savalue.T[0])
    
    # Testing other group
    sashape = s3netcdf.groups["dt"].shape
    data=np.arange(np.prod(sashape)).reshape(sashape)
    s3netcdf["dt","a"] = data.T
    np.testing.assert_array_equal(s3netcdf.query({"variable":"a","idtime":0,"dims":['ndnode']}), data[0])
    
    s3netcdf.cache.delete()
  
  

  

if __name__ == "__main__":
  test_NetCDF2D_2()
  test_NetCDF2D_2_query()
