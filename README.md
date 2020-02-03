# s3-netcdf
Read / write netCDF files to S3


## Installation
```bash
pip install xxx
```
### With AWS S3
```bash
pip3 install awscli
pip install xxx
```


## Usage
```python
from s3netcdf import NetCDF2D

input={
    "name":"test",
    "metadata":{"title":"mytitle"},
    "metadata":{"title":"mytitle"},
    "nc":{"dimensions":["name":"","value":1],"variables":[]},
    "nca":{"name":"a","variables":[]},
    
}

netcdf = NetCDF2D(input)

# Write
netcdf["s","u"]
# TODO : write None,"b"
# TODO : write "s","u"
# TODO : write "t","v"
# TODO : write "ss","e"
# TODO : write "st","e"

# Read

# TODO : read None,"b"
# TODO : read "s","u"
# TODO : read "t","v"
# TODO : read "ss","e"
# TODO : read "st","e"

```

## Testing
### Using pytest


### For developers

```bash
mkdir s3
cd s3
PYTHONPATH=../s3-netcdf/ python3 ../s3-netcdf/test/test_netcdf2d_func.py
```


## Contribution

## License


