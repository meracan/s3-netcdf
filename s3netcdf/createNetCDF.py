import os
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta



from s3netcdf.partitions import getMasterShape,getPartitions
from s3netcdf.netcdf import createNetCDF,getGroupPartition,writeMetadata
from functools import wraps





class NetCDF2D(object):
  def __init__(self, **kwargs):
    self.name = name = kwargs.get('name', "Test") 
    self.folder = folder = kwargs.get('folder', None)
    self.title  = kwargs.get('title', None) 
    self.institution  = kwargs.get('institution', None) 
    self.history  = kwargs.get('history', None) 
    self.references  = kwargs.get('references', None) 
    self.comment  = kwargs.get('comment', None) 
    
    if folder is None: folder = os.getcwd()
    folder = os.path.join(folder, name)
  
    if not os.path.exists(folder):
      os.makedirs(folder)
    
    self.ncPath = os.path.join(folder, "{}.nc".format(name))
    self.ncaPath = os.path.join(folder, "{}.nca".format(name))
    if self.isExist():self.open()
  
  
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
    createNetCDF(self.ncPath,**nc)
    createNetCDF(self.ncaPath,**nca)
    self.open()
    
  def open(self):
    self.nc = Dataset(self.ncPath, "r+")
    self.nca = Dataset(self.ncaPath, "r+")
  
  def close(self):
    self.nc.close()
    self.nca.close()
    
  def findVar(self,vname):
    if vname in self.nc.variables:return self.nc
    if len(self.nc.groups)>0:
      group = self.nc.groups[0]
      if vname in self.nc.variables:return self.nca
    raise Exception("{} does not exist!".format(vname))
  
  def write(self,vname,indices,data):
    src_file = findVar(vname)
    
  
  
  def getGroupPartitions(self):
    if not self.isExist():return
    with Dataset(self.nca, "r") as src_file:
      for group in src_file.groups:
        self.groups[group]=getGroupPartition(src_file,group)
    
  
    
  
  @checkNetCDF
  def writeMetadata(self,**kwargs):
    for filepath in [self.nc,self.nca]:
        writeMetadata(filepath,**kwargs)

  @checkNetCDF
  def write2D(self,vname, data):
    with Dataset(self.nc, "r+") as src_file:
      var = src_file.variables[vname]
      var[:] = data

  @checkNetCDF
  def writePartition(self,group,vname,data,indices):
    data = np.array(data)
    indices = np.array(indices)
    
    

  
  
  def createNCOutput(self,filepath,var):
    None
    # nca = dict(
    #   dimensions = [
    #     dict(name="ntime",value=1),
    #     dict(name="nnode",value=1),
    #     dict(name="nspectra",value=1),
    #     dict(name="nfreq",value=1),
    #     dict(name="ndir",value=1),
    #   ],
    #   groups=[
    #     dict(name="s",strshape=["ntime", "nnode"],variables=vars),
    #     dict(name="t",strshape=["nnode","ntime"],variables=vars),
    #     dict(name="ss",strshape=["ntime", "nspectra", "nfreq", "ndir"],variables=spectravar),
    #     dict(name="ss",strshape=["nspectra","ntime", "nfreq", "ndir"],variables=spectravar)
    #   ]
    # )
    
  
  def createNC(self,**kwargs):
    createNetCDF(self.nc,**kwargs)
    
  def createNCA(self, **kwargs):
    createNetCDF(self.nca,**kwargs)


  
    
  