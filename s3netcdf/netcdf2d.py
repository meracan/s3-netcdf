import os
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta

from s3netcdf.netcdf2d_func import createNetCDF,writeMetadata,getMasterShape
from s3netcdf.netcdf2da import NetCDF2Da
from functools import wraps

class NetCDF2D(object):
  def __init__(self, name,folder,nc,nca,metadata):
    self.netcdfa = {}
    self.name = name
    self.folder = folder
    self.metadata = metadata
    
    if folder is None: folder = os.getcwd()
    self.folder =folder= os.path.join(folder, name)
  
    if not os.path.exists(folder):
      os.makedirs(folder)
    
    self.ncPath = os.path.join(folder, "{}.nc".format(name))
    self.ncaPath = os.path.join(folder, "{}.nca".format(name))
    self.create(nc,nca)
    self.open()
  
  def isExist(self):
    if os.path.exists(self.ncPath) and os.path.exists(self.ncaPath): return True
    return False
  
  def create(self,nc,nca):
    if(self.isExist()):return
    createNetCDF(self.ncPath,folder=self.folder,metadata=self.metadata,**nc)
    createNetCDF(self.ncaPath,folder=self.folder,metadata=self.metadata,**nca)
    
  def open(self):
    self.nc = Dataset(self.ncPath, "r+")
    self.nca = Dataset(self.ncaPath, "r+")
 
    for group in self.nca.groups:
      self.netcdfa[group] = NetCDF2Da(self.folder, self.nca, group,self.name)
  
  def close(self):
    self.nc.close()
    self.nca.close()

  def __getitem__(self, idx):
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("Needs name of group and variable")
    idx = list(idx)
    gname = idx.pop(0)
    if(gname is None):
      vname = idx.pop(1)
      src_file = self.nc
      if not (vname in src_file.variables):raise Exception("Variable does not exist")
      var = self.nc.variables[vname]
      return var[idx]
    else:
      src_file = self.nca
      if not gname in src_file.groups:raise Exception("Group does not exist")
      src_group = src_file.groups[gname]
      netcdfa = self.netcdfa[src_group.name]
      return netcdfa[tuple(idx)]
      
      
  def __setitem__(self, idx,value):
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("Needs name of group and variable")
    idx = list(idx)
    gname = idx.pop(0)
    if(gname is None):
      vname = idx.pop(1)
      src_file = self.nc
      if not (vname in src_file.variables):raise Exception("Variable does not exist")
      var = self.nc.variables[vname]
      var[idx]=value
    else:
      src_file = self.nca
      if not gname in src_file.groups:raise Exception("Group does not exist")
      src_group = src_file.groups[gname]
      netcdfa = self.netcdfa[src_group.name]
      netcdfa[tuple(idx)]=value