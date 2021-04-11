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
s3netcdf.info()
s3netcdf.meta()

# Get group dimensional shape 
s3netcdf.groups["{groupname}"].shape

# Get group dimensional partition shape
s3netcdf.groups["{groupname}"].child

# Get variable's attributes
s3netcdf.groups["{groupname}"].attributes["{variablename}")
```

#### Caching commands
```python
# List partition files locally
s3netcdf.cache.getNCs()

# Clear/Delete all partition files locally
# Warning!
s3netcdf.cache.clearNCs()

# Delete NetCDF locally
# Warning!
# Delete master file and partitions files
s3netcdf.cache.delete()
```


#### S3 commands
```python
# List master and partition files, including metedata
s3netcdf.s3.list()

# Clear/Delete all partition files in S3
# Warning!
s3netcdf.s3.clearNCs()

# Delete NetCDF in S3
# Warning!
# Delete master file and partitions files
s3netcdf.s3.delete()

```

## TODO
- Check operation when index assigning: + - * /

- Fix bench folder and create better performance tests
- Find optimize shape to upload
- travis-ci and encryption keys
- Complete documentation in code
