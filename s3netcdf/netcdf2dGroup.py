import os
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from .netcdf2d_func import createNetCDF,\
  getItemNetCDF,setItemNetCDF,getDataShape,checkValue,getIndices,getPartitions,getMasterIndices,\
  getChildShape,getMasterShape,getSubIndex,parseIndex
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
    Shape of orginal data (e.g. 2,2000)
  master :
    Shape of master file (e.g. 2,2,1,1000)
  child :
    Shape of child file (e.g. 1,1000)
  variablesSetup:
    TODO:Change name "variablesSetup" to "variables"
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
    """
    """
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
    if not vname in self.attributes:raise Exception("Variable does not exist")
    return vname,idx
  
  def getPartitions(self,vname,obj,partitions_only=True):
    """

    """
    shape=self.shape
    masterShape=self.master
    
    if not vname in self.variablesSetup:raise Exception("Variable does not exist in group {}".format(self.name))
    variable=self.variablesSetup[vname]
    
    dimensions=variable['dimensions']
    
    values=[parseIndex(obj.get(dimension[1:],None)) for dimension in dimensions] # Remove n from the dimension name (e.g. ntime=>time). It will look for time in the obj
    
    idx=tuple(values)
    
    indices = getIndices(idx,shape)
    partitions = getPartitions(indices, shape,masterShape)
   
    if partitions_only: return partitions
    return partitions,self,idx


    
  def __run(self,vname,indices,_data=None,_value=None):
    shape = self.shape
    localOnly = self.parent.localOnly
    masterShape = self.master
    s3 = self.parent.s3
    
    partitions = getPartitions(indices, shape,masterShape)
    masterIndices = getMasterIndices(indices,shape,masterShape)

    for part in partitions:
      idata,ipart=getSubIndex(part,shape,masterIndices)
      strpart = "_".join(part.astype(str))
      filepath = os.path.join(self.folderPath, "{}_{}_{}_{}.nc".format(self.parent.name, self.name, vname, strpart))
      
      if _data is not None:
        if not os.path.exists(filepath):
          if self.parent.localOnly:raise Exception("File does not exist. No data was assigned. {}".format(filepath))
          if not self.parent.s3.exists(filepath):raise Exception("File does not exist. No data was assigned. {}".format(filepath))
          self.parent.s3.download(filepath)
          
        d=_data.flatten()
        with Dataset(filepath, "r") as src_file:
          var = src_file.variables[vname]
          d=getItemNetCDF(var,d,ipart,idata)
        _data=d.reshape(_data.shape)
      
      if _value is not None:
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
    return _data
      
  def __subrun(self,vname,idx,value=None):
    shape=self.shape
    indices = getIndices(idx,shape)
    ishape=[len(arr) for arr in indices]
    tmshape,tcshape=getMasterShape(ishape,return_childshape=True,ncSize=100.0)
    if value is not None:
      value=value.reshape(ishape)
    else:
      dataShape = getDataShape(indices)
      data = np.empty(dataShape)
      data=data.reshape(ishape)
      
    
    matrix=[np.arange(idim) for idim in tmshape[:len(ishape)]]
    meshgrid = np.meshgrid(*matrix,indexing="ij")
    tpartitions = np.unique(np.concatenate(np.array(meshgrid).T),axis=0)
    tpartitions=tpartitions.reshape((int(tpartitions.size/len(ishape)),len(ishape)))
    
    for part in tpartitions:

      _local=tuple([slice(i*tcshape[idim],np.minimum((i+1)*tcshape[idim],ishape[idim])) for idim,i in enumerate(part)])
      _indices=tuple([indices[idim][np.arange(i*tcshape[idim],np.minimum(i*tcshape[idim]+tcshape[idim],ishape[idim]))] for idim,i in enumerate(part)])
      if value is not None:
        _value=value[_local]
        self.__run(vname,_indices,_value=_value.flatten())
      else:
        data[_local]=self.__run(vname,_indices,_data=data[_local])
    
    if value is None:
      return data.reshape(dataShape)

  def __getitem__(self, idx):
    """
    Get data based on the query
    """
    vname,idx = self.__checkVariable(idx)
    attributes = self.attributes[vname]

    data=self.__subrun(vname,idx)
    
    if "calendar" in attributes:
      data=data.astype("datetime64[s]")
      # data=num2date(data,units=attributes["units"],calendar=attributes["calendar"])
    
    return data  
    
  def __setitem__(self, idx,value):
    """
    
    """

    vname,idx = self.__checkVariable(idx)
    attributes = self.attributes[vname]
    
    if "calendar" in attributes and isinstance(value[0],datetime):
      value=date2num(value,units=attributes["units"],calendar=attributes["calendar"])

    value= checkValue(value,idx,self.shape)
    self.__subrun(vname,idx,value=value)

  