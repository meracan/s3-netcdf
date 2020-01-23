import pytest
from s3netcdf.createNetCDF import NetCDF2D
import numpy as np



nc=dict(
  dimensions = [
    dict(name="npe",value=3),
    dict(name="nnode",value=1),
    dict(name="ntime",value=1),
    dict(name="nelem",value=1),
  ],
  variables=[
      dict(name="b",type="f4",shape=["nnode"],units="m",standard_name="",long_name=""),
      dict(name="lat",type="f8",shape=["nnode"],units="m",standard_name="",long_name=""),
      dict(name="lng",type="f8",shape=["nnode"],units="m",standard_name="",long_name=""),
      dict(name="elem",type="i4",shape=["nelem"],units="m",standard_name="",long_name=""),
      dict(name="time",type="f8",shape=["ntime"],units="hours since 1970-01-01 00:00:00.0",calendar="gregorian",standard_name="",long_name=""),
    ],
  
)

vars=[
  dict(name="u",type="f4",units="m/s",standard_name="",long_name=""),
  dict(name="v",type="f4",units="m/s",standard_name="",long_name=""),
  dict(name="w",type="f4",units="m/s",standard_name="",long_name=""),
]

spectravar = [
  dict(name="e",type="f4",units="watts",standard_name="",long_name=""),
]

nca = dict(
  dimensions = [
    dict(name="ntime",value=1),
    dict(name="nnode",value=1),
    dict(name="nspectra",value=1),
    dict(name="nfreq",value=1),
    dict(name="ndir",value=1),
  ],
  groups=[
    dict(name="s",strshape=["ntime", "nnode"],variables=vars),
    dict(name="t",strshape=["nnode","ntime"],variables=vars),
    dict(name="ss",strshape=["ntime", "nspectra", "nfreq", "ndir"],variables=spectravar),
    dict(name="ss",strshape=["nspectra","ntime", "nfreq", "ndir"],variables=spectravar)
  ]
)



def test_create():
  nnode=100000
  nelem=100000
 
  lat=np.arange(0, nnode, dtype=np.float64)
  lng=np.arange(0, nnode, dtype=np.float64)-100.0
  elem=np.repeat(np.array([[0,1,2]]),nelem,axis=0)

  netcdf2d=NetCDF2D(metadata)
  netcdf2d.create(nc,nca)
  netcdf2d.write("u",[[0,0]],[0])
  netcdf2d.writeSpectra([[0,0,0,0]],[0])
  netcdf2d.close()
  netcdf2d.createNC(nnode=nnode,nelem=nelem,vars=svars)
  netcdf2d.createNCA(nnode=nnode, vars=vars, ntime=1, nspectra=50, nfreq=33, ndir=36)
  netcdf2d.write2D("lat",lat)
  netcdf2d.write2D("lng", lng)
  netcdf2d.write2D("elem", elem)
  
  
def test_read():
  None

def test_write():
  netcdf2d = NetCDF2D("mytest")
  netcdf2d.writeOutput("u", [[0,0]],[[0]])
  
if __name__ == "__main__":
  # test_create()
  test_write()
  
  