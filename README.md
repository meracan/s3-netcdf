# S3-NetDF
S3-NetCDF is a Python library that stores partitioned NetCDF files on AWS S3.

## Installation
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
#### Create
```python
from s3netcdf import S3NetCDF

input={"name":"testname","cacheLocation":"tempFolder","bucket":"BUCKETNAME","localOnly":True,
  "nca": {
    "metadata":{"title":"testTitle"},
    "dimensions" : {"nnode":1000},
    "groups":{
      "nodes":{"dimensions":["nnode"],"variables":{
          "bed":{"type":"f4", "units":"m" ,"standard_name":"Bed Elevation, m" ,"long_name":"Bed Elevation, m"},
        }
      }
    }
  }
}
S3NetCDF(Input) # <---Create s3netcdf
```
#### Writing/Reading
Once the S3NetCDF is created, the input only needs the name, cacheLocation, and bucket.
```python
from s3netcdf import S3NetCDF
input={"name":"testname","cacheLocation":"tempFolder","bucket":"BUCKETNAME","localOnly":True}

with S3NetCDF(Input) as s3netcdf:         # <---Open s3netcdf
  s3netcdf["nodes","bed"]= np.zeros(1000) # <---Write data
  print(s3netcdf["nodes","bed"])          # <---Read data
```
### Testing and other commands
For other examples and testing scripts: [Docs](test/README.md)

## S3, caching and localOnly
Partition files are saved locally (cached) while reading and writing. The cachedLocation can be changed using the variable: `cacheLocation={path}`. 

The input option `localOnly=True` will ignore all S3 & caching commands. This is used for testing.

The name of the `bucket={str}` in the input if files are uploaded to S3.

## AWS S3 Credentials
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` needs to be saved in environment variables. It is not recommendeded to use the `credentials={AWS_SECRET_ACCESS_KEY=XXX,...}` in the input. 
For more information, check [link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html).
The credentials needs access to AWS S3 commands:`get`, `put` and `delete` (if deleting is required).

## Methodology

S3-NetCDF creates a master file ".nca" from an input object. The input contains AWS S3 info, metadata, dimensions, partition group, variables, etc. Data is stored in the partition files (.nc) (no data is stored in the master file).

Data variables are stored in a partition group. Each partition group has unique variable's dimensions. Multiple variables can be stored under the same partition group (if they have the same dimensions).

The maximum size of partition file is set using the option input `ncSize=10.0`(MB). 
The size is approximative depending on the shape of the array and how it's compressed. 
The partitined files are automatically compressed (~10 smaller). The attribute `least_significant_digit={number}` can be added in the variable object to further reduce file size. 
Remember `f4` and `f8` contains 7 digits 15 digits, respectively. 
S3 http compression (gzip) is not used since partition files are already compressed.

##### Input for writting 
The input for creating a master file contains s3 info, metadata, dimensions, partition group, variables, etc.

Metadata attributes are stored in the `metadata` object. It is recommended to use `title`, `institution`, `source`, `history`, `references`, and `comment`.

Dimensions, groups and variables are stored in the `nca` object.

Input JSON file needs to be converted into a python object `import json; json.loads(filePath)`. 

#### Assigning values
Assigning values to indexed arrays are the same as [numpy](https://docs.scipy.org/doc/numpy/user/basics.indexing.html).

Datetime values needs to be in `np.datetime64`. For example:
```python
timevalue=(np.datetime64(datetime(2001,3,1))+np.arange(np.prod(timeshape))*np.timedelta64(1, 'h'))")
```

#### Query Command Examples
Instead of using python indexing method, the query function can be used with an object as variable.

```python
from s3netcdf import NetCDF2D 
with S3NetCDF(Input) as s3netcdf:
  print(s3netcdf.query({"group":"nodes","variable":"bed"})) # <--- hard code group and variable
  print(s3netcdf.query({"variable":"bed"}))                 # <--- automatically find the variable. If there's multiple variables with the same name in different groups, it will take the smallest smallest/fisrt variable.
```

### License
[License](LICENSE)







