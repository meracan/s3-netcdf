import os
import copy
import numpy as np
from datetime import datetime
from netCDF4 import num2date, date2num,stringtochar,chartostring
from netcdf import NetCDF
from .s3netcdf_func import getItemNetCDF,setItemNetCDF,getDataShape,checkValue,getIndices,getPartitions,getMasterIndices,\
  getChildShape,getMasterShape,getSubIndex,parseIndex,iDim,parseIdx,isQuickSet,isQuickGet
import time
np.seterr(divide='ignore', invalid='ignore', over='ignore')



class S3NetCDFGroup(object):
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
  variables:
    Pre-create dimension and variable array for child file 
    
  
  """
  def __init__(self, parent, name):
    self.parent = parent
    self.src_group = src_group = parent.nca[name]
    self.metadata=metadata=src_group.metadata
    self.folderPath = os.path.join(parent.folder, name)
    self.name = name
    self.shape = list(np.array(metadata['shape'],ndmin=1))
    self.master = list(np.array(metadata['master'],ndmin=1))
    self.child = list(np.array(metadata['child'],ndmin=1))
    self.dimensions = list(np.array(metadata['dims'],ndmin=1))
    self.ndata = len(self.shape)
    self.nmaster = len(self.master)
    self.nchild = len(self.child)
    self.childDimensions=metadata['cdims']
    self.variables = src_group.variables
    
    
  def __checkVariable(self,idx):
    """
    Checking parameters before getting and setting values:
    Needs atleast two axes, name of variable and index value(i.e ["s",:])
    """
    if not isinstance(idx,tuple):raise TypeError("Needs variable")
    idx   = list(idx)
    vname = idx.pop(0)
    idx   = tuple(idx)
    if not vname in self.variables:raise Exception("Variable {} does not exist".format(vname))
    return vname,idx
  
  def getPartitions(self,vname,obj,partitions_only=True):
    """

    """
    if not vname in self.variables:raise Exception("Variable does not exist in group {}".format(self.name))
    
    shape=self.shape
    masterShape=self.master
    variable=copy.deepcopy(self.variables[vname])
    dimensions=variable['dimensions']
    
    values=[parseIdx(obj.get(iDim(dimension),None)) for dimension in dimensions] # Remove n from the dimension name (e.g. ntime=>time). It will look for time in the obj
    for i in range(self.ndata):
      value=values[i]
      if isinstance(value,int):
        if value is not None and value<0:raise Exception("{} needs to be equal or above 0".format(iDim(dimensions[i])))
        if value is not None and value>self.shape[i]:raise Exception("{} needs to be below {}".format(iDim(dimensions[i]),self.shape[i]))
      elif isinstance(value,slice):
        if value.start is not None and value.start<0:raise Exception("{} needs to be equal or above 0".format(iDim(dimensions[i])))
        if value.stop is not None and value.stop<0:raise Exception("{} needs to be equal or above 0".format(iDim(dimensions[i])))
        if value.start is not None and value.start>self.shape[i]:raise Exception("{} needs to be below {}".format(iDim(dimensions[i]),self.shape[i]))
        if value.stop is not None and value.stop>self.shape[i]:raise Exception("{} needs to be below {}".format(iDim(dimensions[i]),self.shape[i]))
      
    idx=tuple(values)
    
    
    indices = getIndices(idx,shape)
    
    partitions = getPartitions(indices, shape,masterShape)

    if partitions_only: return partitions
    return partitions,self,idx,indices

  
  def _getFilepath(self,vname,part):
    strpart = "_".join(part.astype(str))
    return os.path.join(self.folderPath, "{}_{}_{}_{}.nc".format(self.parent.name, self.name, vname, strpart))
  
  def _getFile(self,vname,part):
    localOnly = self.parent.localOnly
    s3 = self.parent.s3
    filepath=self._getFilepath(vname,part)
    if not os.path.exists(filepath):
      if localOnly:raise Exception("File does not exist. No data was assigned. {}".format(filepath))
      if not s3.exists(filepath):
        print("WARNING: File does not exist on S3. Zeroes were assigned to the array. {}".format(filepath))
        return None
      s3.download(filepath)
    return filepath

  def _setFile(self,vname,part):
    localOnly = self.parent.localOnly
    s3 = self.parent.s3
    filepath=self._getFilepath(vname,part)
    if not os.path.exists(filepath):
      if localOnly or not s3.exists(filepath):NetCDF.create(filepath,dimensions=self.childDimensions,variables=copy.deepcopy(self.variables))  
      else:s3.download(filepath)
    return filepath
    
  def __run(self,vname,indices,_data=None,_value=None):
    shape = self.shape
    masterShape = self.master
    partitions = getPartitions(indices, shape,masterShape)
    masterIndices = getMasterIndices(indices,shape,masterShape)

    for part in partitions:
      idata,ipart=getSubIndex(part,shape,masterIndices)
      # GET
      if _data is not None:
        d=_data.flatten()
        filepath=self._getFile(vname,part)
        if filepath is not None:
          with NetCDF(filepath, "r") as netcdf:d=getItemNetCDF(netcdf[vname],d,ipart,idata)
          if not self.parent.localOnly and self.parent.autoRemove:
            try:os.remove(filepath)
            except:print("Already removed")
        _data=d.reshape(_data.shape)
      
      # SET
      if _value is not None:
        filepath=self._setFile(vname,part)
        with NetCDF(filepath, "r+") as netcdf:setItemNetCDF(netcdf[vname],_value,ipart,idata)
        if not self.parent.localOnly:
          self.parent.s3.upload(filepath)
          if self.parent.autoRemove:os.remove(filepath)
    return _data
      
  def __subrun(self,vname,idx,value=None):
    shape=self.shape
    indices = getIndices(idx,shape)
    ishape=[len(arr) for arr in indices]
    tmshape,tcshape=getMasterShape(ishape,return_childshape=True,ncSize=self.parent.memorySize)
    
    if value is not None:
      value=value.reshape(ishape)
    else:
      dataShape = getDataShape(indices)
      ftype=self.variables[vname]['ftype']
      if ftype=='M':ftype='d'
      data = np.zeros(dataShape,dtype=ftype)
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
    
    # if isQuickGet(idx,self.ndata,np.array(self.master)):
    #   data=self._quickGet(vname,list(idx)[0])
    # else:
    data=self.__subrun(vname,idx)
    
    ftype=self.variables[vname]['ftype']
    if ftype=='M':data=data.astype("datetime64[ms]")
    if ftype=='S1':data=chartostring(data)
      
    return data  
    
  def __setitem__(self, idx,value):
    """
    Set data based on the query
    """

    vname,idx = self.__checkVariable(idx)
    if self.variables[vname]['type']=="S":
      value=stringtochar(np.array(value).astype("S{}".format(self.shape[1])))
      
    # if self.variables[vname]['type']!="S":    
    value= checkValue(value,idx,self.shape)
    
    # if isQuickSet(idx,self.ndata,self.master,self.child):
      # self._quickSet(vname,self._getPartIndex(idx),value)
    # else:
    self.__subrun(vname,idx,value=value)
 
    
  def _getPartIndex(self,idx):
    return int(np.floor(list(idx)[0].start/self.child[0]))
    
    
  # def _quickSet(self,vname,partIndex,value):
  #   parts=np.zeros(self.ndata,dtype="int")
  #   parts[0]=partIndex
  #   filepath=self._setFile(vname,parts)
  #   with NetCDF(filepath, "r+") as netcdf:netcdf[vname][:]=value
  #   if not self.parent.localOnly:self.parent.s3.upload(filepath)
  
  # def _quickGet(self,vname,i):
    
  #   parts=np.zeros(self.ndata,dtype="int")
  #   parts[0]=int(np.floor(i/self.child[0]))
  #   index=int(i%self.child[0])
  #   filepath=self._getFile(vname,parts)
  #   with NetCDF(filepath, "r") as netcdf:var = netcdf[vname][index]
  #   return var
    
    # if isinstance(i,int):
    # elif isinstance(i,slice):
    #   parts[0]=0
    #   filepath=self._getFile(vname,parts)
    #   with NetCDF(filepath, "r") as netcdf:var = netcdf[vname][i]
      
    #   return var
  
  # TODO: Transpose  
  #def transpose(self,vanme):
  # if memmap not exist -> check s3 and download (nca2mem), if not give error
  # call mem2memT
  # 
  
  #TODO: get data to memap
  
  #TODO: change code to zarr
  
  # TODO: UploadMemmap
  #def transpose(self,vanme):
  # if memmap not exist -> check s3 and download, if not give error
  #