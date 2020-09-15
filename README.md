# s3-netcdf
S3-NetCDF is a Python library is to store partitioned NetCDF files on AWS S3.
## Installation
```bash
pip install xxx
```
#### From local folder
```bash
git clone https://github.com/meracan/s3-netcdf.git
pip install ./s3-netcdf
```
#### With conda env
```bash
conda create -n s3netcdf python=3.8
conda activate s3netcdf
git clone https://github.com/meracan/s3-netcdf.git
pip install ./s3-netcdf

```
## Usage
#### Writting and Reading
```python
from s3netcdf import NetCDF2D 

# 1. Create object with s3-netcdf info
input={
  "name":"testname",
  "cacheLocation":"tempFolder",
  "bucket":"merac-dev",
  "credentials":{
      "AWS_ACCESS_KEY_ID"="xxx",
      "AWS_SECRET_ACCESS_KEY"="xxx",
      "AWS_DEFAULT_REGION"="xxx"
    },
  "cacheSize":100.0, #MB
  "ncSize":1.0, #MB
  "nca": {
    "metadata":{"title":"testTitle"},
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
        }
      },
      "frames":{"dimensions":["ntime","nnode"],"variables":{
          "u":{"type":"f4", "units":"m" ,"standard_name":"a variable" ,"long_name":"Description of a"}
        }
      }
    }
  }
}

# 2. # Create/Open master file
netcdf2d=NetCDF2D(input) 

# 3. Write
# netcdf2d["{groupname}","{variablename}",{...indices...}]= np.array(...)
netcdf2d["nodes","bed"]= np.zeros(1000)

# 4. Reading
# netcdf2d["{groupname}","{variablename}",{...indices...}]
print(netcdf2d["nodes","bed"])
```
#### Reading Only
```python
from s3netcdf import NetCDF2D 
input={
  "name":"testname",
  "cacheLocation":"tempFolder",
  "bucket":"merac-dev",
  "credentials":{
      "AWS_ACCESS_KEY_ID"="xxx",
      "AWS_SECRET_ACCESS_KEY"="xxx",
      "AWS_DEFAULT_REGION"="xxx"
    },
}
netcdf2d=NetCDF2D(input)
print(netcdf2d["nodes","bed"])
```
## S3, caching and localOnly
Partition files are saved locally (caching) while reading and writing. By default, the `cacheLocation={path}` is the current working directory. 

The input option `cacheSize=100.0` defines the maximum cache size in MB. If exceeded, oldest cached partition files are removed. 

The input option `localOnly=True` will ignore all S3 & caching commands. This is used for testing.

The name of the `bucket={str}` in the input if files are uploaded to S3.

## AWS S3 Credentials
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` can be save in environment variables, instead of hard coding them into the object. 
For more information, check [link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html).
The credentials needs access to AWS S3 commands:`get`, `put` and `delete` (if deleting is required).

## Methodology

S3-NetCDF creates a master file ".nca" from an input object. The input contains AWS S3 info, metadata, dimensions, partition group, variables, etc. Data is stored in the partition files (.nc) (no data is stored in the master file).

Data variables are stored in a partition group. Each partition group has unique variable's dimensions. Multiple variables can be stored under the same partition group (if they have the same dimensions).

The maximum size of partition file (compressed) is set using the option input `ncSize=1.0`(MB). 
The size is approximative depending on the shape of the array and how it's compressed. 
The partitined files are automatically compressed (~10 smaller). The attribute `least_significant_digit={number}` can be added in the variable object to further reduce file size. 
Remember `f4` and `f8` contains 7 digits 15 digits, respectively. 
S3 http compression (gzip) is not used since partition files are already compressed.

##### Input for writting 
The input for creating a master file contains s3 info, metadata, dimensions, partition group, variables, etc.

Metadata attributes are stored in the `metadata` object. It is recommended to use `title`, `institution`, `source`, `history`, `references`, and `comment`.

Dimensions, groups and variables are stored in the `nca` object.

Input JSON file needs to be converted into a python object `import json; json.loads(filePath)`. 

##### Input for reading
The input for opening a master file can be simplified. As a minimum, the input file should contain `name`,`cacheLocation` and `bucket`(if using S3).


#### Assigning values
Assigning values to indexed arrays are the same as [numpy](https://docs.scipy.org/doc/numpy/user/basics.indexing.html).

Datetime values needs to be in `np.datetime64` as seconds. For example:
```python
timevalue=(np.datetime64(datetime(2001,3,1))+np.arange(np.prod(timeshape))*np.timedelta64(1, 'h')).astype("datetime64[s]")
```

#### Query Command Examples
Instead of using python indexing method, the query function can be used with an object as variable.

```python
from s3netcdf import NetCDF2D 
# Create/Open master file
netcdf2d=NetCDF2D(input)
# Get array for all time and node
obj={
  "group":"nodes",
  "variable":"bed",
}
print(netcdf2d.query(obj))
```

The object must have `variable`, and optionally,the group and indices.
The indices depends on the dimension of the group.

For example:
```python
# Extract u variable at Frame=0 for all nodes
obj={
  "variable":"u",
  "time":0,
}

# Extract u variable a Frame=0 for node 0 to 9
obj={
  "variable":"a",
  "time":0,
  "node":"0:10" # index 10 is not included in the array
}

# Extract u variable at Frame=0 for node 0 and 9
obj={
  "variable":"a",
  "time":0,
  "node":"[0,9]"
}

# Extract bed info from node 0 to 9
obj={
  "variable":"bed",
  "node":0:10,
  
}

```
### Testing and other commands
[Docs](test/README.md)

### License
[License](LICENSE)







