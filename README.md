# s3-netcdf
S3-NetCDF is a Python library to read / write NetCDF files to S3. This library partitions large NetCDF files into smaller chunks to retrieve data from s3 cost-effectively.

## Installation
```bash
pip install xxx
```
#### From local folder
```bash
git clone https://github.com/meracan/s3-netcdf.git
pip install -e ./s3-netcdf
```
#### With conda env and testing 
```bash
conda create -n s3netcdf python=3.8
conda activate s3netcdf
git clone https://github.com/meracan/s3-netcdf.git
pip install -e ./s3-netcdf

```

## Methodology

S3-NetCDF creates a master file ".nca" from an input object. The input contains s3 info, metadata, dimensions, partition group, variables, etc. Data is stored in the partition files (.nc) (no data is stored in the master file).

Variables need to be stored in a partition group. Each partition group has unique variable's dimensions. Multiple variables can be stored under the same partition group (if they have the same dimensions).

The maximum size of partition file (umcompressed) is set using the option input `ncSize=1.0`(MB). The size is approximative depending on the shape of the array. The partional files are automatically compressed (~100 smaller). The attribute `least_significant_digit={number}` can be added in the variable object to further reduce file size. Remember `f4` and `f8` contains 7 digits 16 digits, respectively. S3 http compression (gzip) is not used since partition files are already compressed.

##### Input
The input for creating a master file contains s3 info, metadata, dimensions, partition group, variables, etc.

Metadata attributes are stored in the `metadata` object. It is recommended to use `title`, `institution`, `source`, `history`, `references`, and `comment`.

Dimensions, groups and variables are stored in the `nca` object.

Input JSON file needs to be converted into a python object `import json; json.loads(filePath)`. Input example to create a master file: 
```json
{
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":true,
  "bucket":"merac-dev",
  "cacheSize":10.0,
  "ncSize":1.0,
  "metadata":{"title":"title-input1"},
  "nca": {
    "dimensions" : {"npe":3,"nelem":500,"nnode":1000,"ntime":2},
    "groups":{
      "elem":{"dimensions":["nelem","npe"],"variables":{
          "elem":{"type":"i4", "units":"" ,"standard_name":"Elements" ,"long_name":"Connectivity table (mesh elements)"}
        }
      },
      "time":{"dimensions":["ntime"],"variables":{
          "time":{"type":"f8", "units":"hours since 1970-01-01 00:00:00.0","calendar":"gregorian" ,"standard_name":"Datetime" ,"long_name":"Datetime"}
        }
      },
      "nodes":{"dimensions":["nnode"],"variables":{
          "bed":{"type":"f4", "units":"m" ,"standard_name":"Bed Elevation, m" ,"long_name":"Description of data source"},
          "friction":{"type":"f4", "units":"" ,"standard_name":"Bed Friction (Manning's)" ,"long_name":"Description of data source"}
        }
      },
      "s":{"dimensions":["ntime","nnode"],"variables":{
          "a":{"type":"f4", "units":"m" ,"standard_name":"a variable" ,"long_name":"Description of a"}
        }
      },
      "t":{"dimensions":["nnode","ntime"],"variables":{
          "a":{"type":"f4", "units":"m" ,"standard_name":"a variable" ,"long_name":"Description of a"}
        }
      }
    }
  }
}
```

The input for opening  a master file can be simplified. As a minimum, the input file should contain `name`,`cacheLocation` and `bucket`(if using S3).Input example to open a master file:
```json
{
  "name":"input1",
  "cacheLocation":"../s3",
  "bucket":"merac-dev",
  
  "localOnly":true,
  "cacheSize":10.0,
  "ncSize":1.0
}
```

##### S3, caching and localOnly
Partition files are saved locally (caching) while reading and writing. By default, the `cacheLocation={path}` is the current working directory. 

The input option `cacheSize=1.0` defines the maximum cache size in MB. If exceeded, oldest cached partition files are removed. 

The input option `localOnly=True` will ignore all S3 & caching commands. This is used for testing.

The name of the `bucket={str}` in the input if files are uploaded to S3.



## Usage
#### Basic
```python
from s3netcdf import NetCDF2D 
# Create/Open master file
netcdf2d=NetCDF2D(input)

# Writing
netcdf2d["{groupname}","{variablename}",{...indices...}]= np.array(...)

# Reading
netcdf2d["{groupname}","{variablename}",{...indices...}]
```
Assigning values to indexed arrays is the same as [numpy](https://docs.scipy.org/doc/numpy/user/basics.indexing.html). Note: string values was not tested.

#### Commands
```python
# Get information inside the master file
netcdf2d.info()

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

## Testing
```bash
conda install pytest
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
## AWS S3 Credentials
Credentials (for example), AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION needs to be save in environment variables. For more information, check [link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html).

The credentials needs access to `get`, `put` and `delete` (if deleting is required) to the bucket.

## Performance and Benchmark



## TODO
- Revise code on the value parsing side: compare shape, value type etc, Should be in different function and not in dataWrapper.
- Check operation when index assigning: + - * /


- Fix bench folder and create better performance tests
- Find optimize shape to upload
- travis-ci and encryption keys
- Complete documentation in code





