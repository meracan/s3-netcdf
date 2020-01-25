import os
from netCDF4 import Dataset
import numpy as np
from s3netcdf.partitions import getMasterShape,getPartitions,indexMulti


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
      for i,dname in enumerate(dnames):
        dimensions.append(dict(name=dname, value=child[i]))
      
      variable = dict(name=vname, type=src_group.variables[vname].dtype,shape=dnames)
      for attribute in src_group.variables[vname].ncattrs():
        variable[attribute] = getattr(src_group.variables[vname],attribute)
      self.variablesSetup[vname]=dict(dimensions=dimensions,variables=[variable])


  
  def checkSize(self,size):
    if(size>1E+9):raise ValueError("Array too large")
    
  def __getitem__(self, idx):
    idx = list(idx)
    vname = idx.pop(0)
    idx=tuple(idx)
    if not vname in self.variablesSetup:raise Exception("Variable does not exist")
    uniquePartitions, indexPartitions,indexData = getPartitions(idx, self.shape, self.master)
    
    for i,partition in enumerate(uniquePartitions):
      filepath = os.path.join(self.folderPath, "{}_{}_{}_{}_{}.nc".format(self.masterName, self.name, vname, partition[0], partition[1]))
      if not os.path.exists(filepath):
        raise ValueError("File does not exist")
      _index = indexData[indexPartitions==i]
      
      with Dataset(filepath, "r") as src_file:
        var = src_file.variables[vname]
        if(_index.shape[1]==2):
          return var[_index[:,0],_index[:,1]]
        if(_index.shape[1]==3):
          return  var[_index[:,0],_index[:,1],_index[:,2]]
        if(_index.shape[1]==4):
          return var[_index[:,0],_index[:,1],_index[:,2],_index[:,3]]
    

  def __setitem__(self, idx,value):
    
    if not isinstance(idx,tuple):raise TypeError("Needs variable")
    idx = list(idx)
    vname = idx.pop(0)
    idx=tuple(idx)
    if not vname in self.variablesSetup:raise Exception("Variable does not exist")
    
    uniquePartitions, indexPartitions,indexData = getPartitions(idx, self.shape, self.master)
    for i,partition in enumerate(uniquePartitions):
      filepath = os.path.join(self.folderPath, "{}_{}_{}_{}_{}.nc".format(self.masterName, self.name, vname, partition[0], partition[1]))
      
      if not os.path.exists(filepath):
        createNetCDF(filepath,**self.variablesSetup[vname])
      _index = indexData[indexPartitions==i]
      
      if isinstance(value,list) or isinstance(value,np.ndarray):
        # print(_index[:,0],_index[:,1])
        # _value = value[_index[:,0],_index[:,1]].reshape(self.child)
        _value=value
        # TODO: Change this
        print(_value.shape)
        
      else:
        _value=value
      
      
      # print(indexData)
      # print(indexPartitions,indexData)
      # print(self.child)
      with Dataset(filepath, "r+") as src_file:
        # print(src_file.dimensions)
        var = src_file.variables[vname]
        if(_index.shape[1]==2):
          d0=_index[:,0]
          d1=_index[:,1]
          if(np.min(_index[:,0])==np.max(_index[:,0])):
            d0=_index[:,0][0]
          if(np.min(_index[:,0])==0 and np.max(_index[:,0])==self.child[0]-1):
            d0=slice(None,None,None)
          if(np.min(_index[:,1])==0 and np.max(_index[:,1])==self.child[1]-1):
            d1=slice(None,None,None)
            _value = value[i]
          
          # if isinstance(value,list) or isinstance(value,np.ndarray):
            # if(value.shape!=(d0,d1)):
              # value=value.reshape((self.child[0],self.child[1]))
          # print(self.child,_value.shape)
          print(d0,d1)
          var[d0,d1]=_value
        if(_index.shape[1]==3):
          var[_index[:,0],_index[:,1],_index[:,2]]=_value
        if(_index.shape[1]==4):
          var[_index[:,0],_index[:,1],_index[:,2],_index[:,3]]=_value  
        
    # print(indexPartitions,indexData)
    None
    
  
  # def write(self,vname):
    # return GroupArray(self)
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


  

class GroupArray(object):
  def __init__(self, gp):
    self.gp = gp
  
  def __getitem__(self, idx):
  # if isinstance(idx, slice):
  #   start = 0 if slice.start is None else slice.start
  #   end   = gp.shape[0] if slice.end is None else slice.end
  #   step = 1 if slice.step is None else slice.step
  #   (end - start)**
  #   index=np.arange(0,end,step,dtype="i4")
  # elif isinstance( idx, int ) :
      
  # elif isinstance(idx, tuple):
  #   for t in idx:
  #     getType(t,f=False)
  # else:
  #   raise TypeError, "Invalid argument type."  
    
    
            
  #   if isinstance( key, slice ) :
        
  #       return [self[ii] for ii in xrange(*key.indices(len(self)))]
  #   elif isinstance( key, int ) :
  #       if key < 0 : #Handle negative indices
  #           key += len( self )
  #       if key < 0 or key >= len( self ) :
  #           raise IndexError, "The index (%d) is out of range."%key
  #       return self.getData(key) #Get the data from elsewhere
  #   else:
  #       raise TypeError, "Invalid argument type."
    
  #   print(idx,len(idx))
    # print(idx,idx[1].start)
    print(np.array(np.ndindex(3, 2, 1)))
    return None
  def __setitem__(self, idx,value):
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
    _var = src_base.createVariable(var["name"],var["type"], shape,zlib=True,least_significant_digit=3)
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
  master[:],child[:] = getMasterShape(intshape, return_childShape=True, maxSize=1000)
  
def writeVariable(src_file,name,data,indices=None):
  var = src_file.variables[name]
  if indices is None:
    var[:] = data
  else:
    var[indices] = data
