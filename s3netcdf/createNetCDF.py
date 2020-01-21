from netCDF4 import Dataset
from netCDF4 import num2date, date2num
import numpy as np
from datetime import datetime, timedelta


def createNetCDF(filepath,master=False,
                 ntime=0,nnode=1,nelem=1,nspectra=1,nfreq=1,ndir=1,
                 vars=["a"],
                 svars=['e'],
                 type="f8"
                 ):
  # TODO:handle path and name and extension
  # TODO:Create sub folders
  
  with Dataset(filepath, "w") as src_file:
    src_file.title = "File description"
    src_file.institution = "Specifies where the original data was produced"
    src_file.source = "The method of production of the original data"
    src_file.history = "Provides an audit trail for modifications to the original data"
    src_file.references = "Published or web-based references that describe the data or methods used to produce it"
    src_file.comment = "Miscellaneous information about the data or methods used to produce it"
    
    # Dimensions
    src_file.createDimension("npe", 3)  # Nodes per element = 3
    src_file.createDimension("ntime", ntime)  # Number of output time step
    src_file.createDimension("nnode", nnode)  # Number of nodes in the domain
    src_file.createDimension("nelem", nelem)  # Number of elements in the domain
    
    src_file.createDimension("nspectra", nspectra)  # Number of spectra points in the domain
    src_file.createDimension("nfreq", nfreq)
    src_file.createDimension("ndir", ndir)

    shape = ("nnode",)
    sshape = ("nspectra", "nfreq", "ndir")
    
    if(ntime!=0):
      shape=("ntime",) + shape
      sshape = ("ntime",) + sshape
    
    if(master):
      nshape = len(shape)
      src_file.createDimension("nmaster", nshape*2) # 4= for 2D results(temporal,spatial), 4D=8(temporal,spatial,fre,dir)
      src_file.createDimension("nchild", nshape) # 2=2D results, 4=4D results
     
      nsshape = len(sshape)
      src_file.createDimension("nsmaster", nsshape * 2)
      src_file.createDimension("nschild", nsshape)
    
    for var in vars:
      src_file.createVariable(var, type, shape)

    for svar in svars:
      src_file.createVariable(svar, type, sshape)

def writeNetCDF(src_master, indices, data):
  #
  None
  