import os
import sys
from netCDF4 import Dataset,stringtochar,chartostring
from netcdf import NetCDF
import numpy as np
import time
import json
import copy

def createNetCDF(filePath,folder=os.getcwd(),metadata={},dimensions={},variables={},groups={},ncSize=1.0):
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
  
  NetCDF.create(filePath,metadata,dimensions,variables)
  
  with Dataset(filePath, "r+") as netcdf:
    for name in groups:
      group = groups[name]
      if not 'variables' in group:raise Exception("Group needs variables")
      if not 'dimensions' in group:raise Exception("Group needs dimensions")
      dims=group['dimensions']
      variables=group['variables']
      
      # Add dimensions to variable
      for vname in variables:
        variables[vname]['dimensions']=dims
      
      shape=[]
      
      for d in dims:
        if (d==name):raise Exception("group can't have the same name of a dimension")
        if not d in netcdf.dimensions:raise Exception("Dimension {} does not exist".format(d))
        value=netcdf.dimensions[d].size
        shape.append(value)
        
      shape = np.array(shape,dtype="i4")
      master,child=getMasterShape(shape, return_childshape=True, ncSize=ncSize)
      
      group=netcdf.createGroup(name)
      group=netcdf[name]
      
      cdims={}
      for i,d in enumerate(dims):
        cdims[d]=child[i]
      NetCDF._create(group,
        {'cdims':cdims,'shape':shape,'master':master,'child':child,'dims':dims},
        {},
        variables
      )
  
      
      
  
  # if not os.path.isdir(os.path.dirname(filePath)):
  #   os.makedirs(os.path.dirname(filePath), exist_ok=True)
    
  # with Dataset(filePath, "w") as src_file:
  #   # Write metadata
  #   for key in metadata:
  #     value=metadata[key]
  #     if isinstance(value,dict):value=json.dumps(value)
  #     setattr(src_file, key, value)
    
  #   # Write dimensions to NetCDF
  #   for name in dimensions:
  #     src_file.createDimension(name, dimensions[name])
    
  #   # Write variable without a group.  
  #   createVariables(src_file,variables)
    
  #   # Create group
  #   for name in groups:
  #     group = groups[name]
  #     if not 'variables' in group:raise Exception("Group needs variables")
  #     if not 'dimensions' in group:raise Exception("Group needs dimensions")
      
  #     src_group = src_file.createGroup(name)
  #     groupPath = os.path.join(folder, name)
  #     if not os.path.exists(groupPath): os.makedirs(groupPath)

  #     shapeArray=[]
  #     for dimension in group['dimensions']:
  #       if (dimension==name):raise Exception("group can't have the same name of a dimension")
  #       if not dimension in src_file.dimensions:raise Exception("Dimension {} does not exist".format(dimension))
  #       shapeArray.append(len(src_file.dimensions[dimension]))
  #     shapeArray = np.array(shapeArray,dtype="i4")

  #     nshape = len(shapeArray)
  #     src_group.createDimension("nshape", nshape)
  #     src_group.createDimension("nmaster", nshape * 2)
  #     src_group.createDimension("nchild", nshape)
  #     shape = src_group.createVariable("shape", "i4", ("nshape",))
  #     master = src_group.createVariable("master", "i4", ("nmaster",))
  #     child = src_group.createVariable("child", "i4", ("nchild",))

  #     shape[:]=shapeArray
  #     master[:],child[:] = getMasterShape(shapeArray, return_childshape=True, ncSize=ncSize)
  #     src_group.groupDimensions = group['dimensions']
      
  #     createVariables(src_group,group['variables'],group['dimensions'])
      

# def createVariables(src_file,variables,groupDimensions=None):
#   """
#   Create variables in a NetCDF file
  
#   Parameters
#   ----------
#   src_file: NetCDF.Dataset
#   variables:{object}, dict of objects
#     name:str 
#     type:str
#     dimensions:[str],list of str 
#     units:str,optional
#     standard_name:str,optional
#     long_name:str,optional
#     calendar:str,optional
#   groupDimensions: default dimensions for group, optional
  
#   Notes
#   ----------
#   groupDimensions takes priority over dimensions in variables
  
#   """ 
  
#   for name in variables:
#     variable = variables[name]
    
#     if not 'type' in variable:raise Exception("Variable need a type")
#     lsd = variable['least_significant_digit'] if 'least_significant_digit' in variable else None
      
#     if groupDimensions is None:
#       if not 'dimensions' in variable:raise Exception("Variable need dimensions")
#       dimensions = variable['dimensions']
#     else:
#       dimensions=groupDimensions
    
#     _var = src_file.createVariable(name,
#       variable["type"], 
#       dimensions,
#       zlib=True,
#       least_significant_digit=lsd)
#     if "units" in variable:_var.units = variable["units"]
#     if "min" in variable:_var.min = variable["min"]
#     if "max" in variable:_var.max = variable["max"]
#     if "standard_name" in variable:_var.standard_name = variable["standard_name"]
#     if "long_name" in variable:_var.long_name = variable["long_name"]
#     if "calendar" in variable:_var.calendar = variable["calendar"]
    
#     if "data" in variable:
#       if variable['type']=='str':
#         variable['data']=stringtochar(np.array(variable['data']).astype("S{}".format(16))) #TODO: Change 16 to nchar dimension
#       _var[:]=variable['data']




class NpEncoder(json.JSONEncoder):
  """ 
  Encoder to change numpy type to python type.
  This is used for creating JSON object.
  """
  def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return super(NpEncoder, self).default(obj)
  
def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except:
    return False
  return True
  
# def NetCDFSummary(filePath):
#   '''
#   NetCDFSummary outputs metadata,dimensions, variables.
  
#   Parameters
#   ----------
#   filePath:Path

#   Returns
#   -------
#   dict : {metadata,dimensions,variables,groups}
#   Notes
#   -----
#   Only applicable for 1 layer of groups
#   Each group contains variables and dimensions
#   '''
  
#   with Dataset(filePath, "r") as src_file:
#     metadata={}
#     for id in src_file.ncattrs():
#       value=src_file.getncattr(id)
#       if is_json(value):value=json.loads(value)
#       metadata[id]=value
    
#     dimensions={}
#     for id in src_file.dimensions:
#       dimensions[id]=len(src_file.dimensions[id])
    
#     variables =readVariables(src_file)
    
#     groupsByVariable={}
#     groups={}
#     variablesByDimension={}
#     meshMeta={}
#     for id in list(src_file.groups):
#       group = {}
#       group['variables']=readVariables(src_file.groups[id])
#       groupD=src_file.groups[id].cdims
#       if not isinstance(groupD,list):
#         variablesByDimension[groupD]=list(group['variables'].keys())[3:] # [3:] is to remove shape,master and child
#         groupD=[groupD]
      
#       group['dimensions']=groupD
#       groups[id]=group
      
#       # Save variables in vars
#       for key in group['variables'].keys():
#         if key in groupsByVariable:groupsByVariable[key].append(id)
#         else:groupsByVariable[key]=[id]
      
#       # Get mesh information
#       if groupD[0] in ["nnode","nnodes","nelem"]:
#         for key in group['variables'].keys():
#           if key.lower() in ['x','lng','lon','longitude']:
#             meshMeta['x']=key
#           elif key.lower() in ['y','lat','latitude']:
#             meshMeta['y']=key
#           elif key.lower() in ['elem','connectivity','ikle']:
#             meshMeta['elem']=key
    
#     return json.loads(json.dumps(dict(
#       metadata=metadata,
#       dimensions=dimensions,
#       variables=variables,
#       groups=groups,
#       groupsByVariable=groupsByVariable,
#       variablesByDimension=variablesByDimension,
#       meshMeta=meshMeta
#       ),cls=NpEncoder))

# def _summary(netcdf):
#   metadata={}
#   for id in netcdf.ncattrs():
#     value=netcdf.getncattr(id)
#     if is_json(value):value=json.loads(value)
#     metadata[id]=value
  
#   dimensions={}
#   for id in netcdf.dimensions:
#     dimensions[id]=netcdf.dimensions[id].size
  
#   variables={}
#   for vname in netcdf.variables:
#     variable = {}
#     variable['type'] = netcdf.variables[vname].dtype.char
#     variable['dimensions'] = list(netcdf.variables[vname].dimensions)
    
#     for ncattr in netcdf.variables[vname].ncattrs():
#       variable[ncattr]=netcdf.variables[vname].getncattr(ncattr)
#     variables[vname]=variable
#   return variables
  
  
#   # variables =readVariables(src_file)
  


# def readVariables(src):
#   '''
#   Reads and store NetCDF variables (type,dimensions,metadata) into dict.
  
#   Parameters
#   ----------
#   src:netCDF4 object

#   Returns
#   -------
#   dict : {id:{type,dimensions,...metadata}}
#   '''
#   variables={}
#   for id in src.variables:
#     variable = {}
#     dtype=np.dtype(src.variables[id].dtype).name
#     variable['type'] = dtype
#     variable['dimensions'] = list(src.variables[id].dimensions)
    
#     for ncattr in src.variables[id].ncattrs():
#       variable[ncattr]=src.variables[id].getncattr(ncattr)
#     variables[id]=variable
#   return variables
  

# def getVariables(_meta):
#   """
#   TODO
#   """
#   meta=copy.deepcopy(_meta)
#   variables={}
#   for gname in meta['groups']:
#     group=meta['groups'][gname]
#     vars=group['variables']
#     del vars['shape']
#     del vars['master']
#     del vars['child']
#     for vname in vars:
#       variables[vname]=vars[vname]
#   return variables  




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
  
  
  items = 1.0
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
  elif isinstance(idx, int) or isinstance(idx,np.int64):
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
  
  # newarray=np.array((len(shape)-len(array)),dtype="object")
  
  for _,i in enumerate(range(len(array), len(shape))):
    # Note: This loop fills the remainder of the array if not provided
    array.append(np.arange(0, shape[i], dtype="i4"))
  
  return array

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
    if len(indices[i])<len(l):limits.append(indices[i])
    else:limits.append(l)
    
  meshgrid = np.meshgrid(*limits,indexing="ij")
  limits = np.array(meshgrid).T
  limits = np.concatenate(limits)
  
  masterLimits = getMasterIndices(limits.T,shape,masterShape,False)
  
  allPartitions=masterLimits[:, :n]
  uniquePartitions = np.unique(allPartitions,axis=0)
  
  return uniquePartitions


def checkValue(value,idx,shape):
  """
  Check importing/setting values before saving it to NetCDF

  Parameters
  ----------
  value:  ndarray
  idx:  ndarray
  shape: ndarray (1D),dataShape
    Original data shape
  
  Returns
  -------
  out : value,ndarray
  """ 
  dataShape=getDataShape(getIndices(idx,shape))
  if(value is not None):
    if isinstance(value,list):
      value=np.array(value)
    if isinstance(value,(int,float)):
      temp = np.zeros(np.prod(dataShape))+value
      value=temp
    if isinstance(value,str):
      raise Exception("Check Input. Not tested for string")
    if(np.prod(dataShape)!=np.prod(value.shape)):
      raise Exception("Check input. Shape does not match {} and {}".format(dataShape,value.shape))
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


def getItemNetCDF(var,d,ipart,idata):
  """
  Get/Set data from NetCDF4.Dataset using multi-dimensional array indexing
  
  Parameters
  ----------
  var:
  d: ndarray
  ipart:
  idata:
  
  
  
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
  
  if var.attributes['type']=="S":var.isChar=True
  
  all=var[:]
  _d=all[tup]
  if _d.dtype.name!='object':d[idata]=np.squeeze(_d)
  else:d[idata]=_d
  
  return d

def setItemNetCDF(var,d,ipart,idata):
  """ See above
  """
  tup = (0,*ipart.T)[1:]

  if var.attributes['type']=="S":var.isChar=True
    
  all=var[:]
  all[tup]=d[idata]
  var[:]=all
    

  
def getSubIndex(part,shape,masterIndices):
  """ Get partition indices
  Parameters
  ----------
  part:
  shape: ndarray (1D)
  masterIndices:
  """
  n=len(shape)
  idata = np.all(masterIndices[:,:n] == part[None,:], axis=1)
  idata = np.where(idata)[0]
  ipart = masterIndices[idata][:,n:]
  return idata,ipart


def parseIndex(index=None):
  """ 
  
  Parse index query based on string 
  
  Parameters
  ----------
  index:str
  
  Examples
  ---------
  
  """  
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
    
def iDim(dimension):
    return "{}{}".format('i',dimension[1:])

def parseObj(obj,dimensions):
    if not 'variable' in obj:raise Exception("Needs 'variable' in netcdf2d.query")
    newobject={}
    newobject['variable']=obj['variable']
    newobject['group']=obj.get('group',None)
    newobject['dims']=[*obj.get('dims',[])]
    for dim in dimensions:
      idim=iDim(dim)
      newobject[idim]=obj.get(idim,None)
      if not newobject[idim] is None:
        newobject['dims']=[*newobject['dims'],dim]  
    return newobject

def parseIdx(value):
  if value is None:return slice(None,None,None)
  if isinstance(value,str):return parseIndex(value)
  return value
  
def isQuickSet(idx,n,master,child):
  """
  """
  
  lidx=list(idx)
  if not np.all(master[1:n]==1):return False
  if not len(lidx)==1: return False
  item=lidx[0]
  
  if not isinstance(item,slice):return False
  if not (item.stop-item.start)==child[0]: return False
  partIndexStart=int(np.floor(item.start/child[0]))
  partIndexStop=int(np.floor((item.stop-1)/child[0]))
  
  if partIndexStart!=partIndexStop: return False
  return True 
  
# def isQuickSets(idx,n,master,child):
#   """
#   """
  
#   lidx=list(idx)
#   if not np.all(master[1:n]==1):return False
#   if not len(lidx)==1: return False
#   item=lidx[0]
  
#   if not isinstance(item,slice):return False
#   if not (item.stop-item.start)==child[0]: return False
#   partIndexStart=int(np.floor(item.start/child[0]))
#   partIndexStop=int(np.floor((item.stop-1)/child[0]))
  
  
  
  
#   if partIndexStart!=partIndexStop: return False
#   return True 
  
  
def isQuickGet(idx,n,master):
  """
  """
  lidx=list(idx)
  
  if len(lidx)==0: return False
  if len(master[1:n])==0:return False
  if not np.all(master[1:n]==1):return False
  if len(lidx)>2: return False
  if len(lidx)==2 and not lidx[1]==slice(None,None,None):return False
  
  item=lidx[0]
  if isinstance(item,int) or isinstance(item,np.int64) :return True
  else: return False  


# def transform(attributes,value,set=True):
#   """
#   """
#   type=attributes.get("type")
#   min=attributes.get("min")
#   max=attributes.get("max")
#   ftype=attributes.get("ftype","f8")
  
#   maxVs={"uint8":255.0,"uint16":65535.0,"uint32":4294967295.0}
  
#   if not type in maxVs:raise Exception("Not valid type. Needs to be uint8,uint16,uint32")
#   maxV=maxVs[type]
#   f=maxV/(max-min)
#   if set:
#     value=np.clip(value, min, max)
#     return np.rint((value-min)*f).astype(type)
#   else:
#     return (value.astype(ftype)/f)+min