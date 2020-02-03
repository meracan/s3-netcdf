import os
import numpy as np
from netCDF4 import Dataset
from .netcdf2d_func import createNetCDF,dataWrapper


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
  masterName:str,
    Mastername
  
  Attributes
  ----------
  folderPath :
    Folder to save .nc file
  masterName :
    Name of master file
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
  def __init__(self, folder, src_file, name,masterName):
    src_group = src_file[name]
    shape = np.array(src_group.variables["shape"][:])
    master = np.array(src_group.variables["master"][:])
    child = np.array(src_group.variables["child"][:])
    
    self.folderPath = os.path.join(folder, name)
    self.masterName = masterName
    self.name = name
    self.ndata = len(shape)
    self.nmaster = len(master)
    self.nchild = len(child)
    self.shape = shape
    self.master = master
    self.child = child
    
    self.variablesSetup = {}
    
    for vname in src_group.variables:
      if(vname=="shape" or vname=="master" or vname=="child"):continue
      dnames = src_group.variables[vname].dimensions
      dimensions = []
      for i,dname in enumerate(dnames):
        dimensions.append(dict(name=dname, value=child[i]))
      
      variable = dict(name=vname, type=src_group.variables[vname].dtype,shape=dnames)
      for attribute in src_group.variables[vname].ncattrs():
        variable[attribute] = getattr(src_group.variables[vname],attribute)
      self.variablesSetup[vname]=dict(dimensions=dimensions,variables=[variable])

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
    
    array=[]
    def f(part,idata,ipart):
      strpart = part.join("_")
      filepath = os.path.join(self.folderPath, "{}_{}_{}_{}.nc".format(self.masterName, self.name, vname, strpart))
      
      # TODO :check s3, dowload
      if not os.path.exists(filepath):raise Exception("File does not exist")
      
      
      with Dataset(filepath, "r") as src_file:
        var = src_file.variables[vname]
        # TODO: Needs to work for 4 dimensions and not only two
        array.append(np.array(var[ipart[:,0],ipart[:,1]]))
    
    dataWrapper(idx,self.shape,self.master,f)
    
    array=np.concatenate(np.array(array))
    return array
    
    
  def __setitem__(self, idx,value):
    """
    Setting values: dataWrapper gets all partitions and indices, and 
    uses the callback function f() to write data.
    """    
    vname,idx = self.__checkVariable(idx)
    
    def f(part,idata,ipart):
      strpart = part.join("_")
      filepath = os.path.join(self.folderPath, "{}_{}_{}_{}.nc".format(self.masterName, self.name, vname, strpart))
      
      if not os.path.exists(filepath):
        createNetCDF(filepath,**self.variablesSetup[vname])
      
      with Dataset(filepath, "r+") as src_file:
        var = src_file.variables[vname]
        # TODO: Needs to work for 4 dimensions and not only two
        var[ipart[:,0],ipart[:,1]]=value[idata]
      
    dataWrapper(idx,self.shape,self.master,f)