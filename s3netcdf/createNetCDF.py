import os
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta



from s3netcdf.partitions import getMasterShape
from s3netcdf.netcdf import createNetCDF,writeMetadata,GroupPartition
from functools import wraps

class NetCDF2D(object):
  def __init__(self, name,folder,nc,nca,metadata):
    self.groupPartitions = {}
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
  
  def checkNetCDF(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
      if self.isExist():return func(self,*args, **kwargs)
      raise Exception("NetCDF files does not exist. Please create one using the function create")
    return wrapper
  
  def create(self,nc,nca):
    if(self.isExist()):return
    createNetCDF(self.ncPath,folder=self.folder,metadata=self.metadata,**nc)
    createNetCDF(self.ncaPath,folder=self.folder,metadata=self.metadata,**nca)
    
  def open(self):
    self.nc = Dataset(self.ncPath, "r+")
    self.nca = Dataset(self.ncaPath, "r+")
 
    for group in self.nca.groups:
      self.groupPartitions[group] = GroupPartition(self.folder, self.nca, group,self.name)
  
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
      groupPartition = self.groupPartitions[src_group.name]
      return groupPartition[tuple(idx)]
      
      
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
      groupPartition = self.groupPartitions[src_group.name]
      groupPartition[tuple(idx)]=value
      
  
  # def write(self,vname,gname=None):
    
  #   if(gname is None):
  #     src_file = self.nc
  #     if not (vname in src_file.variables):raise Exception("Variable does not exist")
  #     return self.nc.variables[vname]
  #   else:
  #     src_file = self.nca
  #     if not gname in src_file.groups:
  #       raise Exception("Group does not exist")
  #     src_group = src_file.groups[gname]
  #     if not vname in src_group.variables:raise Exception("Variable does not exist")
  #     groupPartition = self.groupPartitions[src_group.name]
  #     return groupPartition.write(vname)
      

 