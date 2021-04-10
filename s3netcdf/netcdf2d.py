import os
import copy
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta
import copy

from .netcdf2d_func import createNetCDF,NetCDFSummary,getMasterShape,parseIndex,parseObj,getVariables,transform
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
  storageClass:str
    STANDARD,STANDARD_IA,ONEZONE_IA
  
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
    self.storageClass=obj.get("storageClass",None)
    self.dynomodb = dynomodb = obj.get("dynomodb",None)
    self.projectId= obj.get("projectId","")
    self.s3prefix=obj.get("s3prefix",None)
    self.squeeze=obj.pop("squeeze",False)
    self.cacheLocation = cacheLocation = obj.get("cacheLocation",os.getcwd)
    self.apiCacheLocation=obj.get("apiCacheLocation",os.getcwd)
    self.folder = folder = os.path.join(cacheLocation, name)
    self.maxPartitions= obj.pop("maxPartitions",10)
    self.ncSize = obj.pop("ncSize",10)
    self.s3 = s3 = S3Client(self,obj.pop("credentials",{}))
    self.cacheSize = obj.pop("cacheSize",10) * 1024**2
    self.cache  = Cache(self)
    self.groups = {}
    self.memorySize=obj.pop("memorySize",20)
    self.verbose= verbose =obj.pop("verbose",False)
    # TODO: Do I need this here?
    self.showProgress=obj.get("showProgress",False)
    self.pbar=None
    overwrite=obj.pop("overwrite",False)
    
    self.isTransform=obj.pop("transform",True) # Transform values from UInt to float
    
    if name is None :raise Exception("NetCDF2D needs a name")
    if not localOnly and bucket is None:raise Exception("Need a s3 bucket")
    if not os.path.exists(folder): os.makedirs(folder,exist_ok = True)
  
    self.ncaPath = ncaPath = os.path.join(folder,"{}.nca".format(name))
    self._meta=None
    
    
    if not os.path.exists(ncaPath):
      if localOnly:
        if verbose:print("Create new .nca from object - localOnly")
        if not "nca" in obj: raise Exception("NetCDF2D needs a nca object, localOnly=True")
        self.create(obj)
      elif s3.exists(ncaPath) and not overwrite:
        if verbose:print("Downloading .nca from S3 - {}".format(ncaPath))
        s3.download(ncaPath)
      else:
        if verbose:print("Create new .nca from object - {} does not exist on S3".format(ncaPath))
        self.create(obj)
        s3.upload(ncaPath)
        if self.dynomodb:s3.insert(ncaPath,self.projectId)
    self.openNCA()
    
  def create(self,obj):
      if not "nca" in obj: raise Exception("NetCDF2D needs a nca object")
      createNetCDF(self.ncaPath,folder=self.folder,ncSize=self.ncSize,**obj["nca"])  
  
  def openNCA(self):
    self.nca = Dataset(self.ncaPath, "r")
    for groupname in self.nca.groups:
      self.groups[groupname] = NetCDF2DGroup(self, self.nca, groupname)
  
  def closeNCA(self):
    self.nca.close()

  def info(self):
    return self.meta
  
  def setTransform(self,transform):
    self.isTransform=transform
    return self
  
  def transform(self,vname,value,set):
    attributes = self.variables[vname]
    return NetCDF2D.transform(attributes,value,set)
  
  @staticmethod
  def transform(attributes,value,set):
    if "min" in attributes and "max" in attributes and attributes['type'] in ["uint8","uint16","uint32"]:
      return transform(attributes,value,set)
    else: 
      raise Exception("Cannot transform type={}, min={},max={})".format(attributes['type'],attributes['min'],attributes['max']))
  
  @property  
  def meta(self):
    if self._meta is None:
      self._meta=NetCDFSummary(self.ncaPath)
    return self._meta

  @property
  def variables(self):
    return getVariables(self.meta)
  
  @property
  def dimensions(self):
    return self.meta['dimensions']
  
  
  
  def uploadedFile(self,gname,vname):
    return os.path.join(self.folder,"uploads","{}.{}.txt".format(gname,vname))
  
  def addUploadedFile(self,gname,vname):
    filepath = self.uploadedFile(gname,vname)
    folder=os.path.dirname(filepath)
    if not os.path.exists(folder):
      os.makedirs(folder,exist_ok=True)
    with open(filepath, "w") as f:f.write("")
    return self.s3.upload(filepath)
  
  def removeUploadedFile(self,gname,vname):
    return self.s3.deleteFile(self.uploadedFile(gname,vname))
    
  def isUploaded(self,gname,vname):
    return self.s3.exists(self.uploadedFile(gname,vname))
  
  
  def getVariables(self):return getVariables(self.meta) #TODO:Depreciate
  
  def setlocalOnly(self,value):
    self.localOnly=value
  
  def getGroupsByVariable(self,vname):
    meta=self.meta
    if not vname in meta['groupsByVariable']:raise Exception("Variable {} does not exist".format(vname))
    return meta['groupsByVariable'][vname]
  
  def getDimensionsByVariable(self,vname):
    meta=self.meta
    if not vname in meta['groupsByVariable']:raise Exception("Variable {} does not exist".format(vname))
    groups=meta['groupsByVariable'][vname]
    dimensions=[]
    for gname in groups:
      dimensions.extend(meta['groups'][gname]['dimensions'])
    
    dimensions=list(set(dimensions))
    return dimensions
  
  def getVariablesByDimension(self,dname):
    meta=self.meta
    variablesByDimension=meta['variablesByDimension']
    if not dname in variablesByDimension:raise Exception("Dimension {} does not exist".format(dname))
    return variablesByDimension[dname]
  
  def getMeshMeta(self):
    meta=self.meta
    return meta['meshMeta']
    
  def getMetaByVariable(self,vname):
    """
    Depreciate
    """
    raise Exception("getMetaByVariable is depreciated")
    # meta=self.meta
    # if not vname in meta['groupsByVariable']:raise Exception("Variable {} does not exist".format(vname))
    # groups=meta['groupsByVariable'][vname]
    # return meta['groups'][groups[0]]['variables'][vname]
  
      
  
  
  def _item_(self,idx):
    if not isinstance(idx,tuple) or len(idx)<2:raise TypeError("groupname and variablename are required, e.g netcdf2d['{groupname}','{variablename}']")
    
    idx = list(idx)
    groupname = idx.pop(0)
    idx = tuple(idx)
    
    groups = self.groups
    if not groupname in groups:raise Exception("Group '{}' does not exist".format(groupname))
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
  
  def query(self,obj,return_dimensions=False,return_indices=False):
    """
      Get data using obj instead of using __getitem__
      This function will search the obj using keys such ash "group","variable" and name of dimensions (e.g. "x","time")
      If "group" does not exist, it will find the "group" based on the name of the variable and with the least amount of partition (e.g "s","t")
    """
    meta=self.meta
    dimensions=meta['dimensions']
    obj=parseObj(obj,dimensions)
    
    # if not 'variable' in obj:raise Exception("Needs 'variable' in query")
    vname=obj['variable']
    gname=obj['group']
    
    if gname is None:
      groups=self.getGroupsByVariable(vname)
      gname=min(groups, key=lambda x: len(self.groups[x].getPartitions(vname,obj)))
    
    partitions,group,idx,indices=self.groups[gname].getPartitions(vname,obj,False)
    
    if len(partitions)>self.maxPartitions:raise Exception("Change group or select smaller query - {} /MaxPartitions is {}".format(len(partitions),self.maxPartitions))
    
    data = group[(vname,*idx)]
    self.cache.clearOldest()
    data= np.squeeze(data) if self.squeeze else data
    
    if return_dimensions and return_indices:return data,meta['groups'][gname]['dimensions'],indices
    if return_dimensions:return data,meta['groups'][gname]['dimensions']
    if return_indices:return data,indices
    return data
