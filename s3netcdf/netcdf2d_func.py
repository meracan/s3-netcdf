import os
from netCDF4 import Dataset
import numpy as np

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
        createNetCDF2Da(src_file,folder,**group)
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
    _var = src_base.createVariable(var["name"],var["type"], shape,zlib=True,least_significant_digit=3)
    if "units" in var:_var.units = var["units"]
    if "standard_name" in var:_var.standard_name = var["standard_name"]
    if "long_name" in var:_var.long_name = var["long_name"]
    if "calendar" in var:_var.calendar = var["calendar"]  

def createNetCDF2Da(src_base,folder,name,variables,strshape):
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
  master[:],child[:] = getMasterShape(intshape, return_childShape=True, maxSize=1000)
  
def writeVariable(src_file,name,data,indices=None):
  var = src_file.variables[name]
  if indices is None:
    var[:] = data
  else:
    var[indices] = data

def getChildShape(shape,dtype="f4",size=1):
  """
  Find new shape based on array size in bytes
  
  Parameters
  ----------
  shape: ndarray(1D).
  dtype: data-type, optional.
    Default is float32.
  size:float,optional.
    Maximum array size (MB)
    Default is 1.
    
  Returns
  -------
  out : ndarray (1D)
  
  Notes
  -----
  Last axis gets priority.
  
  Array size example: 262144 * 4 bytes = 1024**2 (1MB)
  
  Examples
  --------
  >>> a = np.array([2,262144])
  >>> getChildShape(a)
    array([2,1,1,262144])
  >>> b = np.array([2,262145])
  >>> getChildShape(b)
    array([2,2,1,131073])
  """
  
  itemSize = np.dtype(dtype).itemsize
  size = size * 1024.0**2
  
  items = 1
  fileShape = np.ones(len(shape), dtype=np.int)
  for i in range(len(shape) - 1, -1, -1):
    n = shape[i]
    items *= n
    fileSize = items * itemSize
    p = np.int(np.ceil(fileSize / size))
    if (p > 1):
      fileShape[i] = np.int(np.ceil(n * 1.0 / p))
      break
    else:
      fileShape[i] = n
  
  return fileShape


def getMasterShape(shape,return_childshape=False,**kwargs):
  """
  Find new shape based on the child partition. 
  
  Parameters
  ----------
  shape: ndarray(1D).
  return_childshape: bool, optional.
    Default is False.

  Returns
  -------
  out : ndarray (1D)
  
  Notes
  -----
  The number of dimensions/axis will be doubled based on the child dimensions. 
  For example:
  shape      = (10,10)       (x,y)
  childshape = (1,5)         (a,b)    
  out        = (10,2,1,5)    (A,B,a,b)
  The new dimensions (A,B) will be in the order as the child (a,b).
  
  The lenght of out should be equal or larger than shape (x*y => A*B*a*b)
  
  Examples
  --------
  >>> a = np.array([10,10])
  >>> getMasterShape(a)
    array([10,1,1,10])
  >>> b = np.array([10,2,4,8])
  >>> getMasterShape(b)
    array([10,2,2,1,1,1,2,8])
  """

  childshape = getChildShape(shape,**kwargs)
  partitions = np.ceil(np.array(shape) / childshape).astype('int')
  mastershape = np.insert(childshape, 0, partitions)
  if(return_childshape):return mastershape, childshape
  return mastershape