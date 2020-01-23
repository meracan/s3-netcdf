import os
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta



from s3netcdf.partitions import getMasterShape,getPartitions
from functools import wraps

class Shape(object):
  def __init__(self, shape):
    self.data   = shape
    self.master,self.child = getMasterShape(shape, return_childShape=True, maxSize=4)

class Group(object):
  def __init__(self, name,shape):
    self.name = name
    self.shape = Shape(shape)


class NetCDF2D(object):
  def __init__(self, name,folder=None):
    self.initializeStructure(name,folder)
    self.getShapes()
  
  def initializeStructure(self,name,folder):
    self.name=name
    if folder is None: folder = os.getcwd()
    folder = os.path.join(folder, name)
  
    if not os.path.exists(folder):
      os.makedirs(folder)
  
    outputFolder = os.path.join(folder, "output")
    if not os.path.exists(outputFolder):
      os.makedirs(outputFolder)
  
    self.sfolder=SFolder = os.path.join(outputFolder, "s")
    self.tfolder=TFolder = os.path.join(outputFolder, "t")
    self.ssfolder=SSFolder = os.path.join(outputFolder, "ss")
    self.stfolder=STFolder = os.path.join(outputFolder, "st")
    if not os.path.exists(SFolder): os.makedirs(SFolder)
    if not os.path.exists(TFolder): os.makedirs(TFolder)
    if not os.path.exists(SSFolder): os.makedirs(SSFolder)
    if not os.path.exists(STFolder): os.makedirs(STFolder)
  
    self.nc = os.path.join(folder, "{}.nc".format(name))
    self.nca = os.path.join(folder, "{}.nca".format(name))
  
  
  def isExist(self):
    if os.path.exists(self.nc) and os.path.exists(self.nca): return True
    return False
  
  def checkNetCDF(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
      if self.isExist():return func(self,*args, **kwargs)
      raise Exception("NetCDF files does not exist. Please create one using the function create")
    return wrapper
  
  def getShapes(self):
    if not self.isExist():return
    with Dataset(self.nca, "r") as src_file:
      self.dataShape = np.array(src_file.variables['dataShape'][:])
      self.masterShape = np.array(src_file.variables['masterShape'][:])
      self.childShape = np.array(src_file.variables['childShape'][:])
      self.spectraShape = np.array(src_file.variables['spectraShape'][:])
      self.smasterShape = np.array(src_file.variables['smasterShape'][:])
      self.schildShape = np.array(src_file.variables['schildShape'][:])
    
  @checkNetCDF
  def writeMetadata(self,title=None, institution=None, source=None, history=None, references=None, comment=None):
    for filepath in [self.nc,self.nca]:
      with Dataset(filepath, "r+") as src_file:
        if title is not None: src_file.title = title
        if institution is not None: src_file.institution = institution
        if source is not None: src_file.source = source
        if history is not None: src_file.history = history
        if references is not None: src_file.references = references
        if comment is not None: src_file.comment = comment

  @checkNetCDF
  def write2D(self,vname, data):
    with Dataset(self.nc, "r+") as src_file:
      var = src_file.variables[vname]
      var[:] = data

  @checkNetCDF
  def writeOutput(self,vname,indices,data):
    data = np.array(data)
    indices = np.array(indices)
    
    # Spatial
    with Dataset(self.nca, "r+") as src_file:
      var = src_file.variables[vname]
      if(len(var.shape)!=len(indices.shape) or len(var.shape)!=len(data.shape)):
        raise ValueError("Dimensions needs to be the same size")
      
      uniquePartitions, indexPartitions,indexData = getPartitions(indices, self.dataShape, self.masterShape)
      for partition in uniquePartitions:
        filepath=os.path.join(self.sfolder, "{}_s_{}_{}_{}.nc".format(self.name,vname,partition[0],partition[1]))
        # TODO: If file does not exist
        self.createNCOutput(filepath,dict(name="u",units="m/s",standard_name="",long_name=""))
        # TODO: WriteOutput
      
    # Temporal

  
  
  def createNCOutput(self,filepath,var):


    with Dataset(filepath, "w") as src_file:
      src_file.createDimension("ntime", self.childShape[0])  # Number of nodes in the sub-domain
      src_file.createDimension("nnode", self.childShape[1])  # Number of nodes in the sub-domain
      src_file.createDimension("nshape", 2)  # Number of nodes in the sub-domain
      _var = src_file.createVariable("paritionid", "i2", ("nshape",))
      
      _var = src_file.createVariable(var["name"], "f4", ("ntime","nnode",))
      _var.units = var["units"]
      _var.standard_name = var["standard_name"]
      _var.long_name = var["long_name"]


    
    
  
  def createNC(self,nnode=1, nelem=1, vars=None):
    filepath = self.nc
    if vars is None: vars = []
    with Dataset(filepath, "w") as src_file:
      src_file.createDimension("npe", 3)  # Nodes per element = 3
      src_file.createDimension("nnode", nnode)  # Number of nodes in the domain
      src_file.createDimension("nelem", nelem)  # Number of elements in the domain
      
      src_file.createVariable("lat", "f8", ("nnode",))
      src_file.createVariable("lng", "f8", ("nnode",))
      src_file.createVariable("elem", "i4", ("nelem","npe"))
      
      for var in vars:
        _var = src_file.createVariable(var["name"], "f4", ("nnode",))
        _var.units = var["units"]
        _var.standard_name = var["standard_name"]
        _var.long_name = var["long_name"]

  def createNCA(self, ntime=1, nnode=1, nspectra=1, nfreq=1, ndir=1, vars=None):
    filepath = self.nca
    if vars is None: vars = []
    with Dataset(filepath, "w") as src_file:
      
      # Dimensions
      src_file.createDimension("ntime", ntime)  # Number of output time step
      src_file.createDimension("nnode", nnode)  # Number of nodes in the domain
    
      src_file.createDimension("nspectra", nspectra)  # Number of spectra points in the domain
      src_file.createDimension("nfreq", nfreq)
      src_file.createDimension("ndir", ndir)
    
      time = src_file.createVariable("time", "f8", ("ntime",))
      time.units = "hours since 1970-01-01 00:00:00.0"  # TODO: Change based on our needs
      time.calendar = "gregorian"
      time.standard_name = "Datetime"
      time.long_name = "Datetime"
    
      for var in vars:
        _var = src_file.createVariable(var["name"], "f4", ("ntime","nnode",))
        _var.units = var["units"]
        _var.standard_name = var["standard_name"]
        _var.long_name = var["long_name"]
    
      src_file.createVariable("spectra", "f4", ("ntime", "nspectra", "nfreq", "ndir"))
    
      src_file.createDimension("nmaster", 2 * 2)  # 4= for 2D results(temporal,spatial), 4D=8(temporal,spatial,fre,dir)
      src_file.createDimension("noutput", 2)  # 2=2D results, 4=4D results
      dataShape = src_file.createVariable("dataShape", "i4", ("noutput",))
      masterShape = src_file.createVariable("masterShape", "i4", ("nmaster",))
      childShape = src_file.createVariable("childShape", "i4", ("noutput",))
    
      _shape = np.array([ntime, nnode], dtype="i4")
      dataShape[:] = _shape
      masterShape[:], childShape[:] = getMasterShape(_shape, return_childShape=True, maxSize=4)
    
      src_file.createDimension("nsmaster", 4 * 2)
      src_file.createDimension("nsoutput", 4)
      spectraShape = src_file.createVariable("spectraShape", "i4", ("nsoutput",))
      smasterShape = src_file.createVariable("smasterShape", "i4", ("nsmaster",))
      schildShape = src_file.createVariable("schildShape", "i4", ("nsoutput",))
    
      _shape = np.array([ntime, nspectra, nfreq, ndir], dtype="i4")
      spectraShape[:] = _shape
      smasterShape[:], schildShape[:] = getMasterShape(_shape, return_childShape=True, maxSize=4)

