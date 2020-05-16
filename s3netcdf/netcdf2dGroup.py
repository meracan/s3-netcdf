import os
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from .netcdf2d_func import createNetCDF,\
  getItemNetCDF,setItemNetCDF,getDataShape,checkValue,getIndices,getPartitions,getMasterIndices,\
  getChildShape,getMasterShape,getSubIndex
import time
np.seterr(divide='ignore', invalid='ignore', over='ignore')



class NetCDF2DGroup(object):
  """
  A NetCDF class that handles partition files
  
  Parameters
  ----------
  folder : str,path 
      Folder of master, a subfolder is created under here 
  src_file : netCDF4.Dataset
      Master NetCDF2D
  name : str
      Group name (i.e. s,t,ss,st)
  
  
  Attributes
  ----------
  folderPath :
    Folder to save .nc file
  name : str, 
    Group name (i.e. s,t,ss,st)
  ndata :
    Number of axes/dimensions on the original data
  nmaster :
    Number of axes/dimensions on the master file
  nchild :
    Number of axes/dimensions on child. Should be the same to the original data
  shape :
    Shape of orginal data
  master :
    Shape of master file
  child :
    Shape of child file
  variablesSetup:
    TODO:
    Pre-create dimension and variable array for child file 
    
  
  """
  def __init__(self, parent, src_file, name):
    self.src_group = src_group = src_file[name]
    shape = np.array(src_group.variables["shape"][:])
    master = np.array(src_group.variables["master"][:])
    child = np.array(src_group.variables["child"][:])
    
    self.parent = parent
    self.folderPath = os.path.join(parent.folder, name)
    self.name = name
    self.ndata = len(shape)
    self.nmaster = len(master)
    self.nchild = len(child)
    self.shape = shape
    self.master = master
    self.child = child
    self.attributes = self._getAttributes()
    self.variablesSetup = {}
    
    for vname in src_group.variables:
      if(vname=="shape" or vname=="master" or vname=="child"):continue
      dnames = src_group.variables[vname].dimensions
      dimensions = {}
      for i,dname in enumerate(dnames):
        dimensions[dname] = child[i]
      
      variables = {}
      variable = dict(type=src_group.variables[vname].dtype.name,dimensions=dnames)
      for attribute in src_group.variables[vname].ncattrs():
        variable[attribute] = getattr(src_group.variables[vname],attribute)
      variables[vname] = variable
      self.variablesSetup[vname]=dict(dimensions=dimensions,variables=variables)
  
  def _getAttributes(self):
    src_group = self.src_group
    _attributes={}
    for vname in src_group.variables:
      
      variable = src_group.variables[vname]
      attributes = {}
      for attribute in variable.ncattrs():
        attributes[attribute] = getattr(variable,attribute)
      _attributes[vname]=attributes
    return _attributes
  
  def __checkVariable(self,idx):
    """
    Checking parameters before getting and setting values:
    Needs atleast two axes, name of variable and index value(i.e ["s",:])
    """
    if not isinstance(idx,tuple):raise TypeError("Needs variable")
    idx   = list(idx)
    vname = idx.pop(0)
    idx   = tuple(idx)
    if not vname in self.variablesSetup:raise Exception("Variable does not exist")
    return vname,idx
  
  def getPartitions(self,idx):
    shape=self.shape
    masterShape=self.master
    indices = getIndices(idx,shape)
    partitions = getPartitions(indices, shape,masterShape)
    return partitions

  def __getitem__(self, idx):
    """
    
    """
    
    vname,idx = self.__checkVariable(idx)
    attributes = self.attributes[vname]
    
    shape=self.shape
    masterShape=self.master
    
    

    def f(data,indices):
      
      partitions = getPartitions(indices, shape,masterShape)
      masterIndices = getMasterIndices(indices,shape,masterShape)
      
      for part in partitions:
        idata,ipart=getSubIndex(part,shape,masterIndices)
        
        strpart = "_".join(part.astype(str))
        filepath = os.path.join(self.folderPath, "{}_{}_{}_{}.nc".format(self.parent.name, self.name, vname, strpart))
        
        if not os.path.exists(filepath):
          if self.parent.localOnly:raise Exception("File does not exist. No data was assigned. {}".format(filepath))
          if not self.parent.s3.exists(filepath):raise Exception("File does not exist. No data was assigned. {}".format(filepath))
          self.parent.s3.download(filepath)
          
        d=data.flatten()
        with Dataset(filepath, "r") as src_file:
          var = src_file.variables[vname]
          d=getItemNetCDF(var,d,ipart,idata)
        data=d.reshape(data.shape)
        data=np.squeeze(data)
      
      return data
    
    
    indices = getIndices(idx,shape)
    dataShape = getDataShape(indices)
    data = np.empty(dataShape)
    
    if np.prod(dataShape)>1E7:
      for i in range(len(indices[0])):
        # print(i)
        _indices=(np.array(i),indices[1])
        data[i]=f(data[i],_indices)      
      
    else:
      data=f(data,indices)
    
    if "calendar" in attributes:
      data=data.astype("datetime64[s]")
      # data=num2date(data,units=attributes["units"],calendar=attributes["calendar"])
    
    return data
    
    
  def __setitem__(self, idx,value):
    """
    
    """
    localOnly = self.parent.localOnly
    s3 = self.parent.s3
    
    
    vname,idx = self.__checkVariable(idx)
    attributes = self.attributes[vname]
    
    if "calendar" in attributes and isinstance(value[0],datetime):
      value=date2num(value,units=attributes["units"],calendar=attributes["calendar"])
  
    shape=self.shape
    masterShape=self.master
    
    value= checkValue(value,idx,self.shape)

    indices = getIndices(idx,shape)

    def f(_value,_indices):
      partitions = getPartitions(_indices, shape,masterShape)
      masterIndices = getMasterIndices(_indices,shape,masterShape)

      for part in partitions:
       
        idata,ipart=getSubIndex(part,shape,masterIndices)
        strpart = "_".join(part.astype(str))
        filepath = os.path.join(self.folderPath, "{}_{}_{}_{}.nc".format(self.parent.name, self.name, vname, strpart))
 
        if not os.path.exists(filepath):
          if localOnly:createNetCDF(filepath,**self.variablesSetup[vname])  
          else:
            if s3.exists(filepath):s3.download(filepath)
            else:createNetCDF(filepath,**self.variablesSetup[vname])  
        
        with Dataset(filepath, "r+") as src_file:
          var = src_file.variables[vname]
          setItemNetCDF(var,_value,ipart,idata)
        if not localOnly:
          s3.upload(filepath)

    
    if np.prod(value.shape)>1E7:
      nn=len(indices[0])
      pbar=self.parent.pbar
      
      if self.parent.showProgress:
        if pbar is None:
          try:
            from tqdm import tqdm
            pbar =self.parent.pbar= tqdm(total=1)
          except Exception as err:
            import warnings
            warnings.warn("tqdm does not exist")
      
      if pbar: pbar.reset(total=nn)
      for i in range(nn):
        _value=value[i].flatten()
        _indices=(np.array(i),indices[1])
        f(_value,_indices)
        if pbar: pbar.update(1)
    else:
      f(value.flatten(),indices)