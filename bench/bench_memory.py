import numpy as np
from s3netcdf import NetCDF2D
import time

Input = dict(
  name="memory1",
  cacheLocation=r"../s3",
  localOnly=True,

  ncSize=1.0,
  nca = dict(
    metadata=dict(title="memory1"),
    dimensions = dict(
      nnode=1000000,
      ntime=1000000
    ),
    groups=dict(
      g=dict(dimensions=["ntime","nnode"],variables=dict(
        a=dict(type="f4",units="m" ,standard_name="" ,long_name=""),
        )),
    )
  )
)

def bench_1(ncSize,step):
  """
  
   
    
  """
  Input["ncSize"]=ncSize
  netcdf2d=NetCDF2D(Input)
  shape = netcdf2d.groups["g"].shape
  child = netcdf2d.groups["g"].child
  
  end = int(step*2)
  ntime = 1E6
  nnpoint = 1E6
  
  array = np.arange(step*nnpoint)
  total = end*nnpoint
  
  
  tstart = time.time()
  for i in range(0,end,step):
    netcdf2d["g","a",i:i+step] = array+i*step
  tend=time.time()
  print("|{0} | {1} | {2:8.1f}hrs|".format(child,step,(tend-tstart)/total*ntime*nnpoint/60./60.))
  
  netcdf2d.cache.delete()


if __name__ == "__main__":
  bench_1(1.,1)
  bench_1(1.,4)
  bench_1(1.,8)
  bench_1(5.,1)
  bench_1(5.,2)
  bench_1(5.,4)
  bench_1(5.,16)
  bench_1(5.,32)
  bench_1(10.,1)
  bench_1(10.,3)
  bench_1(10.,6)
  bench_1(10.,9)
  bench_1(10.,27)  
  bench_1(100.,1)
  bench_1(100.,27)