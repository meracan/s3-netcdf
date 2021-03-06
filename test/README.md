## Testing
```bash
conda install pytest
mkdir ../s3
pytest
```

For developers and debugging:
```bash
mkdir ../s3
python3 test/test_netcdf2d_func.py
python3 test/test_netcdf2d1.py
python3 test/test_netcdf2d2.py
python3 test/test_netcdf2d3.py
python3 test/test_netcdf2d4.py
python3 test/test_netcdf2d1.py && python3 test/test_netcdf2d2.py && python3 test/test_netcdf2d3.py && python3 test/test_netcdf2d4.py
```

#### General Commands
```python
# Get information inside the master file
netcdf2d.info()
netcdf2d.meta()

# Get group dimensional shape 
netcdf2d.groups["{groupname}"].shape

# Get group dimensional partition shape
netcdf2d.groups["{groupname}"].child

# Get variable's attributes
netcdf2d.groups["{groupname}"].attributes["{variablename}")
```

#### Caching commands
```python
# List partition files locally
netcdf2d.cache.getNCs()

# Clear/Delete all partition files locally
# Warning!
netcdf2d.cache.clearNCs()

# Delete NetCDF locally
# Warning!
# Delete master file and partitions files
netcdf2d.cache.delete()
```


#### S3 commands
```python
# List master and partition files, including metedata
netcdf2d.s3.list()

# Clear/Delete all partition files in S3
# Warning!
netcdf2d.s3.clearNCs()

# Delete NetCDF in S3
# Warning!
# Delete master file and partitions files
netcdf2d.s3.delete()

```

## TODO
- Check operation when index assigning: + - * /

- Fix bench folder and create better performance tests
- Find optimize shape to upload
- travis-ci and encryption keys
- Complete documentation in code
