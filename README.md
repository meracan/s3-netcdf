# s3-netcdf
S3-NetCDF is a Python library to read / write NetCDF files to S3. This library partitions large NetCDF files into smaller chunks to retrieve s3 data cost-effective. 

## Installation
```bash
pip install xxx
```
### From local folder
```bash
git clone https://github.com/meracan/s3-netcdf.git
pip install -e ./s3-netcdf
```

## Usage
```python
# Create and save NetCDF2D object
  netcdf2d=NetCDF2D(Input)
  
  # Get variable shape, create dummy data using arange and save it to netcdf2d:
  #  1 parameter is name of group
  #  2 parameter is name of variable
  elemshape = netcdf2d.getVShape("elem","elem")
  elemvalue = np.arange(np.prod(elemshape)).reshape(elemshape)
  netcdf2d["elem","elem"] = elemvalue
  
  # Read variable and compare with the array above
  np.testing.assert_array_equal(netcdf2d["elem","elem"], elemvalue)
  
  # Write and read "time" variable
  timeshape = netcdf2d.getVShape("time","time")
  timevalue = [datetime(2001,3,1)+n*timedelta(hours=1) for n in range(np.prod(timeshape))]
  netcdf2d["time","time"] = timevalue
  np.testing.assert_array_equal(netcdf2d["time","time"], timevalue)
  
  # Write and read "bed" variable
  bedshape = netcdf2d.getVShape("nodes","bed")
  bedvalue = np.arange(np.prod(bedshape)).reshape(bedshape)
  netcdf2d["nodes","bed"] = bedvalue
  np.testing.assert_array_equal(netcdf2d["nodes","bed"], bedvalue)
  
  # Write and read "a" variable
  sashape = netcdf2d.getVShape("s","a")
  savalue = np.arange(np.prod(sashape)).reshape(sashape)
  netcdf2d["s","a"] = savalue
  np.testing.assert_array_equal(netcdf2d["s","a"], savalue)
  
  # Single int and float will be copied based on the index shape
  netcdf2d["s","a",0,100:200] = 0.0
  np.testing.assert_array_equal(netcdf2d["s","a",0,100:200], np.zeros(100))
  
  tashape = netcdf2d.getVShape("t","a")
  tavalue = np.arange(np.prod(tashape)).reshape(tashape)
  netcdf2d["t","a"] = tavalue
  np.testing.assert_array_equal(netcdf2d["t","a"], tavalue)
```

## Testing
```bash
mkdir ../s3
pytest
```

For developers and debugging:
```bash
mkdir ../s3

PYTHONPATH=../s3-netcdf/ python3 test/test_netcdf2d_func.py
PYTHONPATH=../s3-netcdf/ python3 test/test_netcdf2d1.py
PYTHONPATH=../s3-netcdf/ python3 test/test_netcdf2d2.py
```
## Performance
### Writing
6480 x 3000000 = 1.35 hrs
24 x 365 x 300000 = 8 hrs (10)
24 x 365 x 300000 = 3.5 hrs (50)
24 x 365 x 300000 = 2.8 hrs (100)
### Reading
1 x 3000000 = 0.7 sec

## TODO
- 




