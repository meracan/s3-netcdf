import os
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta

from .netcdf2d_func import createNetCDF,NetCDFSummary,getMasterShape
from .netcdf2dGroup import NetCDF2DGroup


class NetCDF2D(object):
  """
  A NetCDF class
  
  Parameters
  ----------
  object:
    name : str, Name of cdffile
    folder: str,path
    metadata : 
    TODO s3: S3 bucket
    nc:object
      dimensions:[obj]
        name:str
        value:int
      variables:[obj]
        name:str
        type:str
        dimensions:[str],
        etc..
        
    nca:object
      size:float,optional
        Max partition size (MB)
      dimensions:[obj]
        name:str
        value:int
      groups:[obj]
        name:str,
        variables:[obj]
          name:str
          type:str
          dimensions:[str],
          etc..
  
  Attributes
  ----------
  name :str, Name of NetCDF
  folder :path
  ncPath : path
    File contains nodes, connectivity table and static variables 
  ncaPath :path
    Master file, contains information about master and child netcdf files
  netcdfGroups : [NetCDF2DGroup]
    Contains information on different groups (i.e "s","t","ss","st")
    "s" = Spatial oriented
    "t" = Temporal oriented
    "ss" = Spectral Spatial oriented
    "st" = Spectral Temporal oriented
  """  
  def __init__(self, obj):
    if not "name" in obj: raise Exception("NetCDF2D needs a name")
    
    self.netcdfGroups = {}
    self.name = name = obj["name"]
    folder = obj["folder"] if "folder" in obj and obj["folder"] is not None else os.getcwd()
    self.folder = folder= os.path.join(folder, name)
  
    if not os.path.exists(folder): os.makedirs(folder)
    
    self.metadata = obj["metadata"] if "metadata" in obj else dict()
    self.ncPath = os.path.join(folder, "{}.nc".format(name))
    self.ncaPath = os.path.join(folder, "{}.nca".format(name))
    
    # TODO: Check S3, if exist, download 
    if not os.path.exists(self.ncPath) or not os.path.exists(self.ncaPath):
      self.create(obj)
    self.open()
    
  def create(self,obj):
      if not "nc" in obj: raise Exception("NetCDF2D needs a nc object")
      if not "nca" in obj: raise Exception("NetCDF2D needs a nca object")
      createNetCDF(self.ncPath,folder=self.folder,metadata=self.metadata,**obj["nc"])
      createNetCDF(self.ncaPath,folder=self.folder,metadata=self.metadata,**obj["nca"])  
  
  def open(self):
    self.nc = Dataset(self.ncPath, "r+")
    self.nca = Dataset(self.ncaPath, "r+")
 
    for group in self.nca.groups:
      self.netcdfGroups[group] = NetCDF2DGroup(self.folder, self.nca, group,self.name)
  
  def close(self):
    self.nc.close()
    self.nca.close()

  def getSummary(self):
    return {"nc":NetCDFSummary(self.ncPath),"nca":NetCDFSummary(self.ncaPath)}
    

  def getVShape(self,gname,vname):
    if(gname is None):
      src_file = self.nc
      if not (vname in src_file.variables):raise Exception("Variable does not exist")
      var = self.nc.variables[vname]
      return np.array(var.shape)
    else:
      src_file = self.nca
      if not gname in src_file.groups:raise Exception("Group does not exist")
      src_group = src_file.groups[gname]
      netcdfGroup = self.netcdfGroups[src_group.name]
      return netcdfGroup.shape
      
  def __getitem__(self, idx):
    """
    Needs atleast two axes, name of group and variable.
     i.e: ["s","u"]
          [None,"f"]
    """
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("Needs name of group and variable")
    idx = list(idx)
    
    gname = idx.pop(0)
    if(gname is None):
      vname = idx.pop(0)
      src_file = self.nc
      if not (vname in src_file.variables):raise Exception("Variable does not exist")
      var = self.nc.variables[vname]
      if len(idx)==0:
        idx=slice(None,None,None)
      return var[idx]
    else:
      src_file = self.nca
      if not gname in src_file.groups:raise Exception("Group does not exist")
      src_group = src_file.groups[gname]
      netcdfGroup = self.netcdfGroups[src_group.name]
      return netcdfGroup[tuple(idx)]
      
      
  def __setitem__(self, idx,value):
    """
    Needs atleast two axes, name of group and variable.
     i.e: ["s","u"]
          [None,"f"]
    """    
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("Needs name of group and variable")
    idx = list(idx)
    
    gname = idx.pop(0)
    if(gname is None):
      vname = idx.pop(0)
      src_file = self.nc
      if not (vname in src_file.variables):raise Exception("Variable does not exist")
      var = self.nc.variables[vname]
      var[tuple(idx)]=value
    else:
      src_file = self.nca
      if not gname in src_file.groups:raise Exception("Group does not exist")
      src_group = src_file.groups[gname]
      netcdfGroup = self.netcdfGroups[src_group.name]
      netcdfGroup[tuple(idx)]=value