import os
import shutil
import multiprocessing
import numpy as np
from tqdm import tqdm
from slfpy import SLF


def s3netcdf2slf(self,slfPath,**kwargs):
  # Arguments
  variables = kwargs.get('variables')
  startdate = kwargs.get('startDate')
  enddate   = kwargs.get('endDate')
  

  
  
  stepFactor= kwargs.get('stepFactor',1)
  x         = kwargs.get('x',"x")
  y         = kwargs.get('y',"y")
  elem      = kwargs.get('elem',"elem")
  group     = kwargs.get('group',"s")
  title     = kwargs.get('title',self.name)
  
  # Model data
  dimensions  = self.groups[group].dimensions
  timeName    = dimensions[0][1:] # time,dtime,ytime,Dtime
  time        = self.query({"variable":timeName})
  x           = self.query({"variable":x})
  y           = self.query({"variable":y})
  elem        = self.query({"variable":elem})
  s3variables = self.variables
  nnodes      = len(x)
  
  defaultSteps={
    "time":1,
    "dtime":24,
    "ytime":365.25,
    "Dtime":365.25*10,
  }
  
  step        = kwargs.get('step',defaultSteps[timeName])
  stepUnit    = kwargs.get('stepUnit',"h")
  
  isAvg       = True if len(self.groups[group].shape)==3 else False
  
  # Variable metedata
  vnames=[]
  slfvariables=[]
  for vname in variables:
    obj=variables[vname] if isinstance(vname,dict) else {'id':vname}
    vname=obj.get('id')
    variable=s3variables.get(vname)
    name=obj.get('name',vname)
    unit=obj.get('unit',obj.get('units',variable.get('unit',variable.get("units",""))))
    vnames.append(vname)
    if not isAvg:slfvariables.append({"name":name,"unit":unit})
    else:
      slfvariables.append({"name":name+"_MIN","unit":unit})
      slfvariables.append({"name":name+"_MEDIAN","unit":unit})
      slfvariables.append({"name":name+"_MEAN","unit":unit})
      slfvariables.append({"name":name+"_STD","unit":unit})
      slfvariables.append({"name":name+"_MAX","unit":unit})
      
  nvars = len(vnames)
  dtype = "f4"
  fpPath     = os.path.splitext(slfPath)[0]+".npy"
  
  if group=="as":
    # No Temporal data
    startdate  = np.array("1900-01-01",dtype='datetime64')
    usertime   = np.array([startdate])
    nstep      = 1
    shape      = (nvars,5,nnodes) if isAvg else (nvars,nnodes)
    fp         = np.memmap(fpPath,dtype=dtype, mode='w+',shape=shape)
    
    # Download data and save it to memmap
    with tqdm(total=nvars,desc="Downloading") as pbar:
      for k in range(nvars):
        fp[k]=self['as',vnames[k]].T
        pbar.update()
  else:
    # Temporal Info
    startdate = np.array(startdate,dtype='datetime64')
    enddate   = np.array(enddate,dtype='datetime64')
    nstep     = np.floor(((enddate-startdate)/np.timedelta64(step,stepUnit)))
    usertime  = startdate+np.timedelta64(step,stepUnit)*np.arange(0,nstep+1)
    indices   = np.argsort(np.abs(time - usertime[:,np.newaxis]))[:,0]
    nstep     = len(indices)
    usertime  = time[indices]
    
    # Memmap Meta
    iStep      = self.groups[group].child[0]*stepFactor
    shape      = (nstep,nvars,5,nnodes) if isAvg else (nstep,nvars,nnodes)
    fp         = np.memmap(fpPath,dtype=dtype, mode='w+',shape=shape)
    parameters = self.parameters
    
    # Download data and save it to memmap
    with multiprocessing.Pool() as pool:
      with tqdm(total=int(np.ceil(nstep/iStep)),desc="Downloading") as pbar:
        p=pool.imap_unordered(work,[(i,indices,iStep,dtype,shape,parameters,vnames,nvars,group,fpPath) for i in range(0,nstep,iStep)])
        for i, _ in enumerate(p):
          pbar.update()
      
      for i in range(multiprocessing.cpu_count()): # Clean up temporary folders
        shutil.rmtree(os.path.join(parameters['cacheLocation'],str(i)), ignore_errors=True)
  
  # Save data to Selafin
  slf  = SLF()
  slf.addMesh(np.column_stack((x,y)),elem,title=title,var=slfvariables)
  slf.setDatetime(startdate)
  slf.tags['times'] = ((usertime-usertime[0])/ np.timedelta64(1, 's')).astype(np.float32)
  slf.writeHeader(slfPath)
  
  fp = np.memmap(fpPath,dtype=dtype, mode='r', shape=shape)
  
  with tqdm(total=nstep,desc="Writing to Selafin") as t:
      
  
    for i in range(nstep):
      if not isAvg:
        slf.writeFrame(i,fp[i])
      else:
        if group=="as":
          slf.writeFrame(i,fp[:].reshape((nvars*5,nnodes)))
        else:
          array=fp[i].reshape((nvars*5,nnodes))
          slf.writeFrame(i,array)
      t.update()

  os.remove(fpPath)    
  
  
def work(args):
  from .s3netcdf import S3NetCDF
  i,indices,iStep,dtype,shape,parameters,vnames,nvars,group,fpPath=args
  
  nstep      = len(indices)
  j          = np.minimum(nstep,i+iStep)
  size       = j-i
  _indices   = indices[i:j]
  
  parameters = {**parameters,"cacheLocation":os.path.join(parameters['cacheLocation'],str(multiprocessing.current_process()._identity[0]))}
  s3netcdf   = S3NetCDF(parameters)
  fp         = np.memmap(fpPath,dtype=dtype, mode='r+', shape=shape)
  for k in range(nvars):
    fp[i:j,k]=s3netcdf[group,vnames[k],_indices]
  
  return size
  