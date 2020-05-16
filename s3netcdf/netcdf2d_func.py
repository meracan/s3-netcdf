import os
import sys
from netCDF4 import Dataset
import numpy as np
import time

def createNetCDF(filePath,folder=None,metadata=None,dimensions=None,variables=None,groups=None,ncSize=1.0):
  """
  Create typical NetCDF file based on set of variables
  
  Parameters
  ----------
  filePath: str,path
  folder: str,optional.
  metadata:object,optional.
  dimensions:object,optional.
  variables:object,optional.
  groups:object,optional.
  ncSize:object,optional.
  
  Notes
  -----
  Only applicable for 1 layer of groups
  
  
  """  
  if folder is None: folder = os.getcwd()
  if metadata is None: metadata = dict()
  if dimensions is None: dimensions = {}
  if variables is None: variables = {}
  if groups is None: groups = {}
  
  with Dataset(filePath, "w") as src_file:
    # Write metadata
    for key in metadata:
      setattr(src_file, key, metadata[key])
    
    
    
    for name in dimensions:
      src_file.createDimension(name, dimensions[name])
      
    createVariables(src_file,variables)

    for name in groups:
      group = groups[name]
      if not 'variables' in group:raise Exception("Group needs variables")
      if not 'dimensions' in group:raise Exception("Group needs dimensions")
      
      src_group = src_file.createGroup(name)
      groupPath = os.path.join(folder, name)
      if not os.path.exists(groupPath): os.makedirs(groupPath)

      shapeArray=[]
      for dimension in group['dimensions']:
        if (dimension==name):
          raise Exception("group can't have the same name of a dimension")
          
        if not dimension in src_file.dimensions:raise Exception("Dimension does not exist")
        shapeArray.append(len(src_file.dimensions[dimension]))
      shapeArray = np.array(shapeArray,dtype="i4")

      nshape = len(shapeArray)
      src_group.createDimension("nshape", nshape)
      src_group.createDimension("nmaster", nshape * 2)
      src_group.createDimension("nchild", nshape)
      shape = src_group.createVariable("shape", "i4", ("nshape",))
      master = src_group.createVariable("master", "i4", ("nmaster",))
      child = src_group.createVariable("child", "i4", ("nchild",))

      shape[:]=shapeArray
      master[:],child[:] = getMasterShape(shapeArray, return_childshape=True, ncSize=ncSize)
      src_group.groupDimensions = group['dimensions']
      
      createVariables(src_group,group['variables'],group['dimensions'])
      

def createVariables(src_file,variables,groupDimensions=None):
  """
  Create variables in a NetCDF file
  
  Parameters
  ----------
  src_file: NetCDF.Dataset
  variables:{object}, dict of objects
    name:str 
    type:str
    dimensions:[str],list of str 
    units:str,optional
    standard_name:str,optional
    long_name:str,optional
    calendar:str,optional
  groupDimensions: default dimensions for group, optional
  
  Notes
  ----------
  groupDimensions takes priority over dimensions in variables
  
  """ 
  
  for name in variables:
    variable = variables[name]
    if not 'type' in variable:raise Exception("Variable need a type")
    lsd = variable['least_significant_digit'] if 'least_significant_digit' in variable else None
      
    if groupDimensions is None:
      if not 'dimensions' in variable:raise Exception("Variable need dimensions")
      dimensions = variable['dimensions']
    else:
      dimensions=groupDimensions
    
    _var = src_file.createVariable(name,
      variable["type"], 
      dimensions,
      zlib=True,
      least_significant_digit=lsd)
    if "units" in variable:_var.units = variable["units"]
    if "standard_name" in variable:_var.standard_name = variable["standard_name"]
    if "long_name" in variable:_var.long_name = variable["long_name"]
    if "calendar" in variable:_var.calendar = variable["calendar"]  


def NetCDFSummary(filePath):
  '''
  NetCDFSummary outputs metadata,dimensions, variables.
  
  Parameters
  ----------
  filePath:Path

  Returns
  -------
  dict : {metadata,dimensions,variables,groups}
  Notes
  -----
  Only applicable for 1 layer of groups
  Each group contains variables and dimensions
  '''
  
  with Dataset(filePath, "r") as src_file:
    metadata={}
    for id in src_file.ncattrs():
      metadata[id]=src_file.getncattr(id)
    
    dimensions={}
    for id in src_file.dimensions:
      dimensions[id]=len(src_file.dimensions[id])
    
    variables =readVariables(src_file)
    
    vars={}
    groups={}
    for id in list(src_file.groups):
      group = {}
      group['variables']=readVariables(src_file.groups[id])
      group['dimensions']=src_file.groups[id].groupDimensions
      groups[id]=group
      
      # Save variables in vars
      for key in group['variables'].keys():
        if key in vars:
          if isinstance(vars[key],list):vars[key].append(id)
          else:vars[key]=[vars[key],id] 
        else:
          vars[key]=id
      
    
    return dict(metadata=metadata,dimensions=dimensions,variables=variables,groups=groups,vars=vars)


def readVariables(src):
  '''
  Reads and store NetCDF variables (type,dimensions,metadata) into dict.
  
  Parameters
  ----------
  src:netCDF4 object

  Returns
  -------
  dict : {id:{type,dimensions,...metadata}}
  '''
  variables={}
  for id in src.variables:
    variable = {}
    variable['type'] = src.variables[id].dtype.name
    variable['dimensions'] = list(src.variables[id].dimensions)
    
    for ncattr in src.variables[id].ncattrs():
      variable[ncattr]=src.variables[id].getncattr(ncattr)
    variables[id]=variable
  return variables
  
  

def getChildShape(shape,dtype="f4",ncSize=1.0):
  """
  Find new shape based on array size in bytes
  
  Parameters
  ----------
  shape: ndarray(1D).
  dtype: data-type, optional.
    Default is float32.
  ncSize:float,optional.
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
  ncSize = ncSize * 1024.0**2 # 1024**2 = 1MB
  
  items = 1
  fileShape = np.ones(len(shape), dtype=np.int)
  for i in range(len(shape) - 1, -1, -1):
    n = shape[i]
    items *= n
    fileSize = items * itemSize
    p = np.int(np.ceil(fileSize / ncSize))
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
  
  The lenght of output should be equal or larger than shape (x*y => A*B*a*b)
  
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
  


def parseDescriptor(idx,shape,i=0,canTuple=True):
  """
  Parsing descriptor with specific shape
  \
  
  Parameters
  ----------
  idx: slice,int,list,ndarray or tuple.
  shape: ndarray (1D),dataShape
    Original data shape
  i:int,optional.
    Default=0
    Axis position,starts on the left
  canTuple:bool, optional.
    Default is True.
    Can only tuple once. (0,0)=Good;(0,(0))=Bad, will raise an error
    
  Returns
  -------
  out : ndarray or list
  
  Notes
  -----
  First axis gets priority.
  Will not return other axis if not specified,  check "getIndices"
  
  Examples
  --------
  >>> a = [3, 7]
  >>> parseDescriptor(0,a)
    [0]
  >>> parseDescriptor((slice(None,None,None)),a)
    [0 1 2]
  >>> parseDescriptor((slice(None,None,None),slice(None,None,None)),a)
    [[0 1 2],[0 1 2 3 4 5 6]]
  """
  
  if isinstance(idx, slice):
    start = 0 if idx.start is None else idx.start
    stop = shape[i] if idx.stop is None else idx.stop
    step = 1 if idx.step is None else idx.step
    if (start < 0 or stop > shape[i]): raise ValueError("Exceeds limit,start={},stop={},shape[i]={}".format(start,stop,shape[i]))
    return np.arange(start, stop,step, dtype="i4")
  elif isinstance(idx, int):
    if (idx < 0 or idx >= shape[i]): raise ValueError("Exceeds limit,idx={},shape[i]={}".format(idx,shape[i]))
    return np.array([idx],dtype="i4")
  elif isinstance(idx, list) or isinstance(idx, np.ndarray):
    return np.array(idx)
  elif isinstance(idx, tuple):
    if not (canTuple):raise TypeError("Invalid argument type.")
    array=[]
    for j, t in enumerate(idx):
      array.append(parseDescriptor(t,shape,j,canTuple=False))
    return array
  else:
    raise TypeError("Invalid argument type.")  

def getIndices(idx,shape):
  """
  Parsing descriptors and creating a index table based on the shape
  
  Parameters
  ----------
  idx: slice,int,list,ndarray or tuple.
  shape: ndarray (1D),dataShape
    Original data shape
  
  Returns
  -------
  out : ndarray, indices
  
  Notes
  -----
  First axis gets priority.
  Will return other axis if not specified
  
  Examples
  --------
  >>> master = getMasterShape([8, 32768])
  >>> getIndices(0,a)
    [[0],[0 1 2 3 4 5 6]]
  >>> getIndices((slice(None,None,None)),a)
    [[0 1 2],[0 1 2 3 4 5 6]]
  >>> getIndices((slice(None,None,None),slice(None,None,None)),a)
    [[0 1 2],[0 1 2 3 4 5 6]]
  """
  
  array = parseDescriptor(idx,shape,0,canTuple=True)
  if not isinstance(array, list):array=[array]
  for i in range(len(array), len(shape)):
    # Note: This loop fills the remainder of the array if not provided
    array.append(np.arange(0, shape[i], dtype="i4"))
  return np.array(array)

def getMasterIndices(indices,shape,masterShape,expand=True):
  """
  Converts data indices to master indices
  
  Parameters
  ----------
  indices:  ndarray
  shape: ndarray (1D),dataShape
    Original data shape
  masterShape: ndarray (1D)
    Master shape
  
  Returns
  -------
  out : ndarray, indices
  """ 
  
  if isinstance(indices[0],np.ndarray): # Check if 1D or multidimentional
    if(expand):
      indices = np.meshgrid(*indices,indexing="ij")
    
    indices = np.ravel_multi_index(indices, shape)
    indices = indices.flatten()
  return np.array(np.unravel_index(indices, masterShape)).T

def getPartitions(indices,shape,masterShape):
  """
  Find partitions based on indices
  
  Parameters
  ----------
  indices:  ndarray
  shape: ndarray (1D),dataShape
    Original data shape
  masterShape: ndarray (1D)
    Master shape
  raise 
  Returns
  -------
  out : ndarray, partitions
  
  Notes
  -----
  Instead of getting all indices, get only the limits (min, max) of each partition.
  Getting all indices is expensive. 

  """    
  limits=[]
  n = len(shape)
  
  for i,step in enumerate(masterShape[n:]):
    _min =np.min(indices[i])
    _max = np.max(indices[i])
    l=np.arange(_min,_max,step)
    
    if(np.all(l!=_max)):
      l=np.append(l,_max)
    limits.append(l)
  
  
  meshgrid = np.meshgrid(*limits,indexing="ij")
  limits = np.array(meshgrid).T
  
  limits = np.concatenate(limits)
  masterLimits = getMasterIndices(limits.T,shape,masterShape)
  allPartitions=masterLimits[:, :n]
  uniquePartitions = np.unique(allPartitions,axis=0)
  return uniquePartitions


def checkValue(value,idx,shape):
  """
  A few checks when setting data

  Parameters
  ----------
  value:  ndarray
  idx:  ndarray
  shape: ndarray (1D),dataShape
    Original data shape
  
  Returns
  -------
  out : value
  """ 
  dataShape=getDataShape(getIndices(idx,shape))
  if(value is not None):
    if isinstance(value,list):
      value=np.array(value)
    #   value = value.flatten()
    # if isinstance(value,np.ndarray):
    #   value = value.flatten()
    if isinstance(value,(int,float)):
      temp = np.zeros(np.prod(dataShape))+value
      value=temp
    if isinstance(value,str):
      raise Exception("Check Input. Not tested for string")
    if(np.prod(dataShape)!=np.prod(value.shape)):
      # TODO, try repeat row...
      raise Exception("Check input. Shape does not match {} and {}".format(dataShape,value.shape))
  # print(value)
  return value

def getDataShape(indices):
  """
  Create new data shape based on the selected index array (from idx)

  Parameters
  ----------
  indices:  ndarray
  
  Returns
  -------
  out : ndarray
  """ 
  dataShape=[]
  for i in range(len(indices)):
    dataShape.append(len(indices[i]))
  dataShape=tuple(dataShape)
  return dataShape

  

def getItemNetCDF(*args,**kwargs):
  return _getset_ItemNetCDF(*args,**kwargs,get=True)

def setItemNetCDF(*args,**kwargs):
  return _getset_ItemNetCDF(*args,**kwargs,get=False)




def _getset_ItemNetCDF(var,d,ipart,idata,get=True):
  """
  Get or set data from NetCDF4.Dataset using multi-dimensional array indexing
  
  Parameters
  ----------
  var:
  d: ndarray (1D)
    
  ipart:
    
  idata:
  
  get:
  
  
  Returns
  -------
  d : ndarray
  
  Note 
  ----
  The initial approach was simply use multi-dimensional array indexing
  using the following procudure,d[idata]=np.squeeze(var[(0,*ipart.T)[1:]]).
  This does not work well with netCDF4.Dataset since it does not handle 
  numpy multi-dimensional arrays indexing.
  
  For more information on "Indexing Multi-dimensional arrays",
  https://docs.scipy.org/doc/numpy/user/basics.indexing.html
  
  Solution
  --------
  Save netcdf array to numpy, use multi-dimensional array indexing on the numpy 
  array,convert back to netcdf if necessary
  """
  
  tup = (0,*ipart.T)[1:]
  if(get):
    all=var[:]
    d[idata]=np.squeeze(all[tup])
  else:
    all=var[:]
    all[tup]=d[idata]
    var[:]=all
    
  return d
  
  
def getSubIndex(part,shape,masterIndices):
  """
  """
  n=len(shape)
  idata = np.all(masterIndices[:,:n] == part[None,:], axis=1)
  idata = np.where(idata)[0]
  ipart = masterIndices[idata][:,n:]
  return idata,ipart


def parseIndex(index):
  try:
    if index is None:
      value=slice(None,None,None)  
    else:
      index=str(index)
      
      if ":" in index:
        start,end=index.split(":")
        if start=="":start=None
        else: start=int(start)
        if end=="":end=None
        else: end=int(end)
        value=slice(start,end,None)
      elif "[" in  index:
        index=index.replace("[","").replace("]","")
        index=index.split(",")
        value=list(map(lambda x:int(x),index))
      else:
        value=int(index)
    return value
  except Exception as err:
    raise Exception("Format needs to be \"{int}\" or \":\" or \"{int}:{int}\" or \"[{int},{int}]\"")