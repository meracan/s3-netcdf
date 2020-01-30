import pytest
from s3netcdf.netcdf2d_func import createNetCDF,\
  writeMetadata,createVariables,createGroupPartition,writeVariable,\
  getChildShape,getMasterShape
  
import numpy as np