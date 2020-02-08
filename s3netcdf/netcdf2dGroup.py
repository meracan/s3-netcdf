import os
import numpy as np
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from .netcdf2d_func import createNetCDF,dataWrapper,getItemNetCDF,setItemNetCDF
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
      # TODO: clean this section
      if(vname=="shape" or vname=="master" or vname=="child"):continue
      dnames = src_group.variables[vname].dimensions
      dimensions = []
      for i,dname in enumerate(dnames):
        dimensions.append(dict(name=dname, value=child[i]))
      
      variable = dict(name=vname, type=src_group.variables[vname].dtype.name,dimensions=dnames)
      for attribute in src_group.variables[vname].ncattrs():
        variable[attribute] = getattr(src_group.variables[vname],attribute)
      self.variablesSetup[vname]=dict(dimensions=dimensions,variables=[variable])
  
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
    

  def __getitem__(self, idx):
    """
    Getting values: dataWrapper gets all partitions and indices, and 
    uses the callback function f() to extract data.
    """
    
    vname,idx = self.__checkVariable(idx)
    attributes = self.attributes[vname]
    
    def f(part,idata,ipart,data,uvalue):
      strpart = "_".join(part.astype(str))
      filepath = os.path.join(self.folderPath, "{}_{}_{}_{}.nc".format(self.parent.name, self.name, vname, strpart))
      
      if not os.path.exists(filepath):
        if self.parent.localOnly:raise Exception("File does not exist. No data was assigned")
        if not self.parent.s3.exists(filepath):raise Exception("File does not exist. No data was assigned")
        self.parent.s3.download(filepath)
        
      d=data.flatten()
      with Dataset(filepath, "r") as src_file:
        var = src_file.variables[vname]
        d=getItemNetCDF(var,d,ipart,idata)
      data=d.reshape(data.shape)
      return np.squeeze(data)
          
        
    data=dataWrapper(idx,self.shape,self.master,f)
    
    if "calendar" in attributes:
      data=num2date(data,units=attributes["units"],calendar=attributes["calendar"])
    
    return data
    
    
  def __setitem__(self, idx,value):
    """
    Setting values: dataWrapper gets all partitions and indices, and 
    uses the callback function f() to write data.
    """
    localOnly = self.parent.localOnly
    autoUpload = self.parent.autoUpload
    s3 = self.parent.s3
    
    
    vname,idx = self.__checkVariable(idx)
    attributes = self.attributes[vname]
    
    # if isinstance(value,np.ndarray):
      # value = value.flatten()
      
    
    # TODO value : handles int,float, etc. 
    
    if "calendar" in attributes:
      value=date2num(value,units=attributes["units"],calendar=attributes["calendar"])
    
    
    # start = time.time() 
    def f(part,idata,ipart,data,uvalue):
      
      strpart = "_".join(part.astype(str))
      filepath = os.path.join(self.folderPath, "{}_{}_{}_{}.nc".format(self.parent.name, self.name, vname, strpart))
      
      if not os.path.exists(filepath):
        if localOnly:createNetCDF(filepath,**self.variablesSetup[vname])  
        else:
          if s3.exists(filepath):s3.download(filepath)
          else:createNetCDF(filepath,**self.variablesSetup[vname])  
      
      # start = time.time()  
      with Dataset(filepath, "r+") as src_file:
        var = src_file.variables[vname]
        setItemNetCDF(var,uvalue,ipart,idata)
      # print(time.time() - start)
      if not localOnly and autoUpload:
        s3.upload(filepath)
      
      
    dataWrapper(idx,self.shape,self.master,f,value)
    # print(time.time() - start)