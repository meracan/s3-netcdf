import os
import boto3
from botocore.errorfactory import ClientError
from time import sleep
import time
class S3Client(object):
  """
  Interface to communicate with S3
  
  """
  def __init__(self, parent,credentials={}):
    """
    parent=netcdf2d object
    credentials={
      AWS_ACCESS_KEY_ID=xxx,
      AWS_SECRET_ACCESS_KEY=xxx,
      AWS_DEFAULT_REGION=xxx
    }
    """
    
    self.s3 = boto3.client('s3',**credentials)
    self.s3prefix=parent.s3prefix
    self.parent = parent
    self.bucket = parent.bucket
  
  def _gets3path(self,filepath):
    s3path = os.path.relpath(filepath,self.parent.cacheLocation)
    if self.s3prefix:
      s3path = self.s3prefix + "/" + s3path
    return s3path
  
  def list(self):
    Prefix=self.parent.name
    if self.s3prefix:
      Prefix= self.s3prefix + "/" + Prefix
    
    page_iterator = self.s3.get_paginator('list_objects_v2').paginate(Bucket=self.bucket, Prefix=Prefix)
    files=[]
    for page in page_iterator:
      if "Contents" in page:
        files=files+page["Contents"]
    return files
  
    
  def clearNCs(self):
  
    files = self.list()
    if len(files)==0:return True
    
    ncs = [{"Key":file["Key"]} for file in files if os.path.splitext(file['Key'])[1]==".nc"]
    self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects":ncs})
    return True
  
    
  def delete(self):
    files = self.list()
    if len(files)==0:return True
    
    keys = [{"Key":file["Key"]} for file in files]
    response=self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects":keys})
    return True

  def exists(self,filepath):
    s3path = self._gets3path(filepath)
    try:
      self.s3.head_object(Bucket=self.bucket, Key=s3path)
      return True
    except ClientError as e:
      return False
  
  def loop(self,callback):
    """
    loop to trying to get something from the web
    """
    try_loops=5
    while try_loops > 0:
      try:
        callback()
        try_loops=0
      except Exception as err:
        sleep(5)
        try_loops -= 1
        if try_loops <1:raise err    

  def download(self,filepath):
    bucket = self.bucket
    s3path = self._gets3path(filepath)
    
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder): os.makedirs(folder)
    self.s3.download_file(bucket, s3path, filepath)
    # print("Download - {} - {}".format(filepath,time.time() - start))
    # self.loop(lambda:self.s3.download_file(bucket, s3path, filepath))

    
  def upload(self,filepath):
    bucket = self.bucket
    s3path = self._gets3path(filepath)
    self.s3.upload_file(filepath, bucket, s3path)
    # self.loop(lambda:self.s3.upload_file(filepath, bucket, s3path))
