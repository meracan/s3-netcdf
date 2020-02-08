import os
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta

from .netcdf2d_func import createNetCDF,NetCDFSummary,getMasterShape
from .netcdf2dGroup import NetCDF2DGroup
from .s3client import S3Client
from .cache import Cache

class NetCDF2D(object):
  """
  A NetCDF class
  
  Parameters
  ----------
  object:
    name : str, Name of cdffile
    folder: str,path
    metadata : 
    
    bucket: Name of S3 bucket
    autoUpload:bool
    localOnly:bool
      To include or ignore s3 storage
      Default:True
    ncSize:float,optional
        Max partition size (MB)   
    nca:object
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
  localOnly:bool
  bucket:str
  autoUpload:bool
  metadata:obj
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
    self.name = name = obj["name"] if "name" in obj else None
    self.localOnly = localOnly = obj["localOnly"] if "localOnly" in obj else True
    self.bucket = bucket = obj["bucket"] if "bucket" in obj else None
    self.autoUpload = autoUpload = obj["autoUpload"] if "autoUpload" in obj else True
    cacheLocation = obj["cacheLocation"] if "cacheLocation" in obj and obj["cacheLocation"] is not None else os.getcwd()
    self.metadata = obj["metadata"] if "metadata" in obj else dict()
    cacheSize = obj["cacheSize"] if "cacheSize" in obj else 10
    self.ncSize = obj["ncSize"] if "ncSize" in obj else 1
    
    if name is None :raise Exception("NetCDF2D needs a name")
    if not localOnly and bucket is None:raise Exception("Need a s3 bucket")
    
    self.cacheLocation = cacheLocation
    self.folder = folder = os.path.join(cacheLocation, name)
    if not os.path.exists(folder): os.makedirs(folder)
    
    self.netcdfGroups = {}
    self.s3 = s3 = S3Client(self)
    self.cache  = Cache(self)
    self.cacheSize = cacheSize * 1024**2
    
    self.ncaPath = ncaPath = os.path.join(folder,"{}.nca".format(name))
    
    if not os.path.exists(ncaPath):
      if localOnly:self.create(obj)
      else:
        if s3.exists(ncaPath):s3.download(ncaPath)
        else:
          self.create(obj)
          s3.upload(ncaPath)
    
    self.open()
    
  
    
  def create(self,obj):
      if not "nca" in obj: raise Exception("NetCDF2D needs a nca object")
      createNetCDF(self.ncaPath,folder=self.folder,metadata=self.metadata,ncSize=self.ncSize,**obj["nca"])  
  
  def open(self):
    self.nca = Dataset(self.ncaPath, "r+")
 
    for group in self.nca.groups:
      self.netcdfGroups[group] = NetCDF2DGroup(self, self.nca, group)
  
  def close(self):
    self.nca.close()

  def getSummary(self):
    return NetCDFSummary(self.ncaPath)
    

  def getVShape(self,gname,vname):
    netcdfGroups = self.netcdfGroups
    if not gname in netcdfGroups:raise Exception("Group does not exist")
    return self.netcdfGroups[gname].shape
  
  def getVAttributes(self,gname,vname):
    netcdfGroups = self.netcdfGroups
    if not gname in netcdfGroups:raise Exception("Group does not exist")
    return self.netcdfGroups[gname].attributes[vname]
  
  
  def __getitem__(self, idx):
    """
    Needs atleast two axes, name of group and variable.
     i.e: ["s","u"]
          [None,"f"]
    """
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("Needs name of group and variable")
    idx = list(idx)
    
    gname = idx.pop(0)
    src_file = self.nca
    if not gname in src_file.groups:raise Exception("Group does not exist")
    src_group = src_file.groups[gname]
    netcdfGroup = self.netcdfGroups[src_group.name]
    data = netcdfGroup[tuple(idx)]
    self.cache.clearOldest()
    return data
  
      
      
  def __setitem__(self, idx,value):
    """
    Needs atleast two axes, name of group and variable.
     i.e: ["s","u"]
          [None,"f"]
    """    
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("Needs name of group and variable")
    idx = list(idx)
    
    gname = idx.pop(0)
    src_file = self.nca
    if not gname in src_file.groups:raise Exception("Group does not exist")
    src_group = src_file.groups[gname]
    netcdfGroup = self.netcdfGroups[src_group.name]
    netcdfGroup[tuple(idx)]=value
    self.cache.clearOldest()