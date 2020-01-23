import os
from netCDF4 import Dataset
import numpy as np
from s3netcdf.partitions import getMasterShape,getPartitions


class GroupPartition(object):
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
      for dname in dnames:
        dimensions.append(dict(name=dname, value=len(src_file.dimensions[dname])))
      
      variable = dict(name=vname, type=src_group.variables[vname].dtype,shape=dnames)
      for attribute in src_group.variables[vname].ncattrs():
        variable[attribute] = getattr(src_group.variables[vname],attribute)
      self.variablesSetup[vname]=dict(dimensions=dimensions,variables=[variable])
    
    
  
  def write(self,vname):
    return CC()
    # data = np.array(data)
    # indices = np.array(indices)
    # if (len(self.shape) != len(indices.shape) or len(indices) != len(data)):
    #     raise ValueError("Dimensions needs to be the same size")
    # uniquePartitions, indexPartitions, indexData = getPartitions(indices, self.shape, self.master)
    #
    # for partition in uniquePartitions:
    #   filepath = os.path.join(self.folderPath, "{}_{}_{}_{}_{}.nc".format(self.masterName, self.name, vname, partition[0], partition[1]))
    #   if not os.path.exists(filepath):
    #     createNetCDF(filepath,**self.variablesSetup[vname])
    #   # TODO:
    # print(indexPartitions,indexData)

class CC(object):
  def __getitem__(self, idx):
    print(idx)
    return None
# class NamedRows(np.ndarray):
#   def __new__(cls, *args, **kwargs):
#     obj = np.asarray(*args, **kwargs).view(cls)
#     # obj.__row_name_idx = dict((n, i) for i, n in enumerate(rows))
#     return obj
#
#   def __getitem__(self, idx):
#     return super(NamedRows, self).__getitem__(idx)

def createNetCDF(filepath,folder=None,metadata=None,dimensions=None,groups=None,variables=None):
    if dimensions is None: dimensions = []
    if groups is None: groups = []
    if variables is None: variables = []
    if metadata is None: metadata = dict()
    with Dataset(filepath, "w") as src_file:
      writeMetadata(src_file,**metadata)
      # Dimensions
      for dimension in dimensions:
        src_file.createDimension(dimension['name'], dimension['value'])
      
      # Groups and Variables
      for group in groups:
        createGroupPartition(src_file,folder,**group)
      createVariables(src_file,variables)

def writeMetadata(src_file,title=None, institution=None, source=None, history=None, references=None, comment=None):
  if title is not None: src_file.title = title
  if institution is not None: src_file.institution = institution
  if source is not None: src_file.source = source
  if history is not None: src_file.history = history
  if references is not None: src_file.references = references
  if comment is not None: src_file.comment = comment

def createVariables(src_base,variables,strshape=None):
  for var in variables:
    if strshape is None and var["shape"] is None:raise Exception("Variable needs a shape")
    shape = strshape if strshape is not None else var["shape"]
    _var = src_base.createVariable(var["name"], var["type"], shape)
    if "units" in var:_var.units = var["units"]
    if "standard_name" in var:_var.standard_name = var["standard_name"]
    if "long_name" in var:_var.long_name = var["long_name"]
    if "calendar" in var:_var.calendar = var["calendar"]  

def createGroupPartition(src_base,folder,name,variables,strshape):
  src_group = src_base.createGroup(name)
  groupPath = os.path.join(folder, name)
  if not os.path.exists(groupPath): os.makedirs(groupPath)
  
  createVariables(src_group,variables,strshape)
  
  intshape=[]
  for ishape in strshape:
    intshape.append(len(src_base.dimensions[ishape]))
  intshape = np.array(intshape,dtype="i4")
  
  nshape = len(intshape)
  src_group.createDimension("nshape", nshape)
  src_group.createDimension("nmaster", nshape * 2)
  src_group.createDimension("nchild", nshape)
  shape = src_group.createVariable("shape", "i4", ("nshape",))
  master = src_group.createVariable("master", "i4", ("nmaster",))
  child = src_group.createVariable("child", "i4", ("nchild",))
  
  shape[:]=intshape
  master[:],child[:] = getMasterShape(intshape, return_childShape=True, maxSize=4)
  
def writeVariable(src_file,name,data,indices=None):
  var = src_file.variables[name]
  if indices is None:
    var[:] = data
  else:
    var[indices] = data
