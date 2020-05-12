import os
from tqdm import tqdm
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta

from .netcdf2d_func import createNetCDF,NetCDFSummary,getMasterShape,parseIndex
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
  
  ncPath : path
    File contains nodes, connectivity table and static variables 
  ncaPath :path
    Master file, contains information about master and child netcdf files
  groups : [NetCDF2DGroup]
    Contains information on different groups (i.e "s","t","ss","st")
    "s" = Spatial oriented
    "t" = Temporal oriented
    "ss" = Spectral Spatial oriented
    "st" = Spectral Temporal oriented
  """  
  def __init__(self, obj):
    self.name = name = obj.get("name",None)
    self.localOnly = localOnly = obj.get("localOnly",True)
    self.bucket = bucket = obj.get("bucket",None)
    self.s3prefix=obj.get("prefix",None)
    cacheLocation = obj.get("cacheLocation",os.getcwd)
    
    cacheSize = obj.get("cacheSize",10)
    self.ncSize = obj.get("ncSize",1)
    
    if name is None :raise Exception("NetCDF2D needs a name")
    if not localOnly and bucket is None:raise Exception("Need a s3 bucket")
    
    self.cacheLocation = cacheLocation
    self.folder = folder = os.path.join(cacheLocation, name)
    if not os.path.exists(folder): os.makedirs(folder)
    
    showProgress=obj.get("showProgress",False)
    self.pbar = tqdm(total=1) if showProgress else None 
    
    self.groups = {}
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
      
      createNetCDF(self.ncaPath,folder=self.folder,ncSize=self.ncSize,**obj["nca"])  
  
  def open(self):
    self.nca = Dataset(self.ncaPath, "r+")
    for groupname in self.nca.groups:
      self.groups[groupname] = NetCDF2DGroup(self, self.nca, groupname)
  
  def close(self):
    self.nca.close()

  def info(self):
    return NetCDFSummary(self.ncaPath)
  
  def setlocalOnly(self,value):
    self.localOnly=value
    
  def _item_(self,idx):
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("groupname and variablename are required")
    
    idx = list(idx)
    groupname = idx.pop(0)
    idx = tuple(idx)
    
    groups = self.groups
    if not groupname in groups:raise Exception("Group does not exist")
    group = groups[groupname]
    return group,idx
  
  def __getitem__(self, idx):
    """
      ["{groupname}","{variablename}",{...indices...}]
    """
    group,idx=self._item_(idx)
    data = group[idx]
    self.cache.clearOldest()
    return data
      
  def __setitem__(self, idx,value):
    """
      ["{groupname}","{variablename}",{...indices...}]=np.array()
    """    
    group,idx=self._item_(idx)
    group[idx]=value
    self.cache.clearOldest()
    
  def query(self,obj):
    """
      Get data using obj instead of using __getitem__
      This function will search the obj using keys such ash "group","variable" and name of dimensions (e.g. "x","time")
    """
    groups = self.groups
    
    groupname=obj["group"]
    vname=obj["variable"]
    
    if not groupname in groups:raise Exception("Group does not exist")
    group = groups[groupname]
    
    if not vname in group.variablesSetup:raise Exception("Variable does not exist")
    variable=group.variablesSetup[vname]
    
    dimensions=variable['dimensions']
    
    values=[parseIndex(obj.get(dimension[1:],None)) for dimension in dimensions] # Remove n from the dimension name (e.g. ntime=>time). It will look for time in the obj
    
    idx=tuple(values)
    partitions=group.getPartitions(idx)
    if len(partitions)>10:raise Exception("Change group or select smaller arraysize")
    data = group[(vname,*idx)]
    self.cache.clearOldest()
    return data
    
    