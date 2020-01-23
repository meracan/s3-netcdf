from netCDF4 import Dataset
import numpy as np
from s3netcdf.partitions import getMasterShape,getPartitions

def createNetCDF(filepath,dimensions=None,groups=None,variables=None):
    if dimensions is None: dimensions = []
    if groups is None: groups = []
    if variables is None: variables = []
    with Dataset(filepath, "w") as src_file:
      # Dimensions
      for dimension in dimensions:
        src_file.createDimension(dimension['name'], dimension['value'])
      
      # Groups and Variables
      for group in groups:
        createGroupPartition(src_file,**group)
      createVariables(src_file,variables)

def createVariables(src_base,variables,strshape=None):
  for var in variables:
    if strshape is None and var["shape"] is None:raise Exception("Variable needs a shape")
    shape = strshape if strshape is not None else var["shape"]
    _var = src_base.createVariable(var["name"], var["type"], shape)
    if "units" in var:_var.units = var["units"]
    if "standard_name" in var:_var.standard_name = var["standard_name"]
    if "long_name" in var:_var.long_name = var["long_name"]
    if "calendar" in var:_var.calendar = var["calendar"]  

def createGroupPartition(src_base,name,strshape):
  intshape=[]
  for ishape in strshape:
    intshape.append(len(src_base.dimensions[ishape]))
  intshape = np.array(intshape,dtype="i4")

# TODO : create folders
## self.stfolder=STFolder = os.path.join(outputFolder, "st")
    # if not os.path.exists(SFolder): os.makedirs(SFolder
  src_group = src_base[name]
  nshape = len(intshape)
  src_group.createDimension("nshape", nshape)
  src_group.createDimension("nmaster", nshape * 2)
  src_group.createDimension("nchild", nshape)
  shape = src_group.createVariable("shape", "i4", ("nshape",))
  master = src_group.createVariable("master", "i4", ("nmaster",))
  child = src_group.createVariable("schildShape", "i4", ("nsoutput",))
  
  shape[:]=intshape
  master[:],child[:] = getMasterShape(intshape, return_childShape=True, maxSize=4)


class GroupPartition(object):
  def __init__(self, name,shape,master,child):
    self.name = name
    self.ndata = len(shape)
    self.nmaster = len(master)
    self.nchild = len(child)
    self.shape  = shape
    self.master = master
    self.child  = child
    
def getGroupPartition(src_base,name):
  src_group = src_base[name]
  shape = np.array(src_group.variables["shape"][:])
  master = np.array(src_group.variables["master"][:])
  child = np.array(src_group.variables["child"][:])
  return GroupPartition(name,shape,master,child)
  
def writeMetadata(filepath,title=None, institution=None, source=None, history=None, references=None, comment=None):
  with Dataset(filepath, "r+") as src_file:
    if title is not None: src_file.title = title
    if institution is not None: src_file.institution = institution
    if source is not None: src_file.source = source
    if history is not None: src_file.history = history
    if references is not None: src_file.references = references
    if comment is not None: src_file.comment = comment

def writeVariable(src_file,name,data,indices=None):
  var = src_file.variables[name]
  if indices is None:
    var[:] = data
  else:
    var[indices] = data

def writePartition(folder,name,gp,vname,data,indices,dimensions):
  if(len(gp.shape)!=len(indices.shape) or len(gp.shape)!=len(data.shape)): 
    raise ValueError("Dimensions needs to be the same size")
  uniquePartitions, indexPartitions,indexData = getPartitions(indices, gp.shape, gp.master)
  for partition in uniquePartitions:
    filepath=os.path.join(folder, "{}_{}_{}_{}_{}.nc".format(name,gp.name,vname,partition[0],partition[1]))
  if not os.path.exists(filepath):
    options = dict(
      dimensions=[
        dict(name="ntime",value=1),
        dict(name="nnode",value=1),
      ],
    variables=[
      dict(name="u",type="f4",units="m/s",standard_name="",long_name=""),
    ])
    createNetCDF(filepath,**options)
  
  writeVariable(filepath,vname,data,indices)
  
  