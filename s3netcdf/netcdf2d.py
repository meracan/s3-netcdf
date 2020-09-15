import os

from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta
import copy

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
    squeeze : squeeze get data - mainly used for testing
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
    Contains information on different groups or folders (i.e "s","t")
    Each group contains different variables but with the same dimensions
  """  
  def __init__(self, _obj):
    obj = copy.deepcopy(_obj)
    self.name = name = obj.get("name",None)
    self.localOnly = localOnly = obj.get("localOnly",True)
    self.bucket = bucket = obj.get("bucket",None)
    self.s3prefix=obj.get("prefix",None)
    self.squeeze=obj.pop("squeeze",False)
    self.cacheLocation = cacheLocation = obj.get("cacheLocation",os.getcwd)
    self.folder = folder = os.path.join(cacheLocation, name)
    self.maxPartitions= obj.pop("maxPartitions",10)
    self.ncSize = obj.pop("ncSize",10)
    self.s3 = s3 = S3Client(self,obj.pop("credentials",{}))
    self.cacheSize = obj.pop("cacheSize",10) * 1024**2
    self.cache  = Cache(self)
    self.groups = {}
    self.verbose= verbose =obj.pop("verbose",False)
    # TODO: Do I need this here?
    self.showProgress=obj.get("showProgress",False)
    self.pbar=None
    
    if name is None :raise Exception("NetCDF2D needs a name")
    if not localOnly and bucket is None:raise Exception("Need a s3 bucket")
    if not os.path.exists(folder): os.makedirs(folder)
  
    self.ncaPath = ncaPath = os.path.join(folder,"{}.nca".format(name))
    
    if not os.path.exists(ncaPath):
      if localOnly:
        if verbose:print("Create new .nca from object - localOnly")
        self.create(obj)
      elif s3.exists(ncaPath):
        if verbose:print("Downloading .nca from S3 - {}".format(ncaPath))
        s3.download(ncaPath)
      else:
        if verbose:print("Create new .nca from object - {} does not exist on S3".format(ncaPath))
        self.create(obj)
        s3.upload(ncaPath)
    self.openNCA()
    
  def create(self,obj):
      if not "nca" in obj: raise Exception("NetCDF2D needs a nca object")
      createNetCDF(self.ncaPath,folder=self.folder,ncSize=self.ncSize,**obj["nca"])  
  
  def openNCA(self):
    self.nca = Dataset(self.ncaPath, "r+")
    for groupname in self.nca.groups:
      self.groups[groupname] = NetCDF2DGroup(self, self.nca, groupname)
  
  def closeNCA(self):
    self.nca.close()

  def info(self):
    return NetCDFSummary(self.ncaPath)
    
  def meta(self):
    return NetCDFSummary(self.ncaPath)

  def setlocalOnly(self,value):
    self.localOnly=value
  
  def getVariables(self,groupname):
    meta=self.meta()
    if not groupname in meta['groups']:raise Exception("Group does not exist")
    data=meta['groups'][groupname]['variables']
    del data['shape']
    del data['master']
    del data['child']
    return data

  def _item_(self,idx):
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("groupname and variablename are required, e.g netcdf2d['{groupname}','{variablename}']")
    
    idx = list(idx)
    groupname = idx.pop(0)
    idx = tuple(idx)
    
    groups = self.groups
    if not groupname in groups:raise Exception("Group does not exist")
    group = groups[groupname]
    return group,idx
  
  def __getitem__(self, idx):
    """
      netcdf2d["{groupname}","{variablename}",{...indices...}]
    """
    group,idx=self._item_(idx)
    data = group[idx]
    self.cache.clearOldest()
    data= np.squeeze(data) if self.squeeze else data
    return data
      
  def __setitem__(self, idx,value):
    """
      netcdf2d["{groupname}","{variablename}",{...indices...}]=np.array()
    """    
    group,idx=self._item_(idx)
    group[idx]=value
    self.cache.clearOldest()

  def query(self,obj):
    """
      Get data using obj instead of using __getitem__
      This function will search the obj using keys such ash "group","variable" and name of dimensions (e.g. "x","time")
      If "group" does not exist, it will find the "group" based on the name of the variable and with the least amount of partition (e.g "s","t")
    """
    if not 'variable' in obj:raise Exception("Needs 'variable' in query")
    vname=obj['variable']
    gname=obj.get('group',None)
    
    if gname is None:
      meta=self.meta()
      if not vname in meta['vars']:raise Exception("Variable does not exist")
      gname=meta['vars'][vname]
    
      # If  multiple group, get group with minimum partitions
      if isinstance(gname,list):
        gname=min(gname, key=lambda x: len(self.groups[x].getPartitions(vname,obj)))
    
    partitions,group,idx=self.groups[gname].getPartitions(vname,obj,False)
    if len(partitions)>self.maxPartitions:raise Exception("Change group or select smaller query - (admin) maxPartitions can be changed")
    
    data = group[(vname,*idx)]
    self.cache.clearOldest()
    data= np.squeeze(data) if self.squeeze else data
    return data