import os
import copy
# from netCDF4 import Dataset
# from netCDF4 import num2date, date2num
from netcdf import NetCDF
import numpy as np
from datetime import datetime, timedelta
import copy

from .s3netcdf_func import createNetCDF,getMasterShape,parseIndex,parseObj
from .s3netcdfGroup import S3NetCDFGroup
from .s3client import S3Client
from .cache import Cache

class S3NetCDF(object):
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
    
    overwrite=obj.pop("overwrite",False)
    
    
    if name is None :raise Exception("NetCDF2D needs a name")
    if not localOnly and bucket is None:raise Exception("Need a s3 bucket")
    if not os.path.exists(folder):os.makedirs(folder,exist_ok = True)
  
    self.ncaPath = ncaPath = os.path.join(folder,"{}.nca".format(name))
    
    
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
      createNetCDF(self.ncaPath,ncSize=self.ncSize,**obj["nca"])  
  
  def openNCA(self):
    self.nca=NetCDF(self.ncaPath,"r")
    for groupname in self.nca.groups:
      self.groups[groupname] = S3NetCDFGroup(self, self.nca, groupname)
  
  def closeNCA(self):
    self.nca.close()

  @property  
  def obj(self):return self.nca.obj

  @property
  def dimensions(self):return self.nca.obj['dimensions']

  @property
  def variables(self):return self.nca.allvariables

  def getGroupsByVariable(self,vname):
    groupsByVariable=self.nca.obj['groupsByVariable']
    if not vname in groupsByVariable:raise Exception("Variable {} does not exist".format(vname))
    return groupsByVariable[vname]
    
  def getVariablesByDimension(self,dname):
    variablesByDimension=self.nca.obj['variablesByDimension']
    if not dname in variablesByDimension:raise Exception("Dimension {} does not exist".format(dname))
    return variablesByDimension[dname]
    
  def setlocalOnly(self,value):
    self.localOnly=value
  
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
    if isinstance(idx,str):return self.groups[idx] 
    group,idx=self._item_(idx)
    data = group[idx]
    data= np.squeeze(data) if self.squeeze else data
    return data
      
  def __setitem__(self, idx,value):
    """
      netcdf2d["{groupname}","{variablename}",{...indices...}]=np.array()
    """    
    group,idx=self._item_(idx)
    group[idx]=value
  
  def query(self,obj,return_dimensions=False,return_indices=False):
    """
      Get data using obj instead of using __getitem__
      This function will search the obj using keys such ash "group","variable" and name of dimensions (e.g. "x","time")
      If "group" does not exist, it will find the "group" based on the name of the variable and with the least amount of partition (e.g "s","t")
    """
    dimensions=self.dimensions
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
    data= np.squeeze(data) if self.squeeze else data
    
    if return_dimensions and return_indices:return data,self.obj['groups'][gname]['dimensions'],indices
    if return_dimensions:return data,self.obj['groups'][gname]['dimensions']
    if return_indices:return data,indices
    return data
