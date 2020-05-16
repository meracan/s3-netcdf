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
pip install ./s3-netcdf

```

## Methodology

S3-NetCDF creates a master file ".nca" from an input object. The input contains s3 info, metadata, dimensions, partition group, variables, etc. Data is stored in the partition files (.nc) (no data is stored in the master file).

Variables need to be stored in a partition group. Each partition group has unique variable's dimensions. Multiple variables can be stored under the same partition group (if they have the same dimensions).

The maximum size of partition file (umcompressed) is set using the option input `ncSize=1.0`(MB). The size is approximative depending on the shape of the array. The partional files are automatically compressed (~100 smaller). The attribute `least_significant_digit={number}` can be added in the variable object to further reduce file size. Remember `f4` and `f8` contains 7 digits 16 digits, respectively. S3 http compression (gzip) is not used since partition files are already compressed.

##### Input - Writting
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
  
  "nca": {
    "metadata":{"title":"title-input1"},
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

##### Input - Reading
The input for opening  a master file can be simplified. As a minimum, the input file should contain `name`,`cacheLocation` and `bucket`(if using S3).
Input example to open a master file:
```json
{
  "name":"input1",
  "cacheLocation":"../s3",
  "bucket":"merac-dev",
}
```

## Usage
#### Basic
```python
from s3netcdf import NetCDF2D 
# Create/Open master file
netcdf2d=NetCDF2D(input)

# Writing
# netcdf2d["{groupname}","{variablename}",{...indices...}]= np.array(...)
netcdf2d["s","a"]= np.zeros((2,1000))

# Reading
# netcdf2d["{groupname}","{variablename}",{...indices...}]
print(netcdf2d["s","a"])
```
Assigning values to indexed arrays is the same as [numpy](https://docs.scipy.org/doc/numpy/user/basics.indexing.html).

Datetime values needs to be in `np.datetime64` as seconds. For example:
```python
timevalue=(np.datetime64(datetime(2001,3,1))+np.arange(np.prod(timeshape))*np.timedelta64(1, 'h')).astype("datetime64[s]")
```

#### Using object instead of getitem
```python
from s3netcdf import NetCDF2D 
# Create/Open master file
netcdf2d=NetCDF2D(input)
# Get array for all time and node
obj={
  "group":"s",
  "variable":"a",
}
print(netcdf2d.query(obj))
```

The object must have `group`, `variable`, and optionally, the indices. The indices depends on the dimension of the group. For example:
```python
# Extract time=0 for all nodes
obj={
  "group":"s",
  "variable":"a",
  "time":0,
}

# Extract variable a time=0 for node 0 to 9
obj={
  "group":"s",
  "variable":"a",
  "time":0,
  "node":"0:10" # index 10 is not included in the array
}

# Extract variable at time=0 for node 0 and 9
obj={
  "group":"s",
  "variable":"a",
  "time":0,
  "node":"[0,9]"
}

# Extract bed info from node 0 to 9
obj={
  "group":"nodes",
  "variable":"bed",
  "node":0:10,
  
}

```


##### S3, caching and localOnly
Partition files are saved locally (caching) while reading and writing. By default, the `cacheLocation={path}` is the current working directory. 

The input option `cacheSize=1.0` defines the maximum cache size in MB. If exceeded, oldest cached partition files are removed. 

The input option `localOnly=True` will ignore all S3 & caching commands. This is used for testing.

The name of the `bucket={str}` in the input if files are uploaded to S3.


## AWS S3 Credentials
Credentials (for example), AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION needs to be save in environment variables. 
For more information, check [link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html).
The credentials needs access to `get`, `put` and `delete` (if deleting is required) to the bucket.

### Testing and other commands
[Docs](test/README.md)

### License
[License](LICENSE)







