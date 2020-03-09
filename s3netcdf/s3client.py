import os
import boto3
from botocore.errorfactory import ClientError
s3 = boto3.client('s3')

    
class S3Client(object):
  """
  Interface to communicate with S3
  
  """
  def __init__(self, parent):
    self.parent = parent
    self.bucket = parent.bucket
  
  def _gets3path(self,filepath):
    s3path = os.path.relpath(filepath,self.parent.cacheLocation)
    return s3path
  
  def list(self):
    try:
      page_iterator = s3.get_paginator('list_objects_v2').paginate(Bucket=self.bucket, Prefix=self.parent.name)
      files=[]
      for page in page_iterator:
        if "Contents" in page:
          files=files+page["Contents"]
      return files
    except ClientError as e:
      raise Exception(e)
    
  def clearNCs(self):
    try:
      files = self.list()
      if len(files)==0:return True
      
      ncs = [{"Key":file["Key"]} for file in files if os.path.splitext(file['Key'])[1]==".nc"]
      s3.delete_objects(Bucket=self.bucket, Delete={"Objects":ncs})
      return True
    except ClientError as e:
      raise Exception(e)
    
  def delete(self):
    try:
      files = self.list()
      if len(files)==0:return True
      
      keys = [{"Key":file["Key"]} for file in files]
      s3.delete_objects(Bucket=self.bucket, Delete={"Objects":keys})
      return True
    except ClientError as e:
      raise Exception(e)
  
  def exists(self,filepath):
    s3path = self._gets3path(filepath)
    
    try:
      s3.head_object(Bucket=self.bucket, Key=s3path)
      return True
    except ClientError as e:
      return False
      

  def download(self,filepath):
    bucket = self.bucket
    s3path = self._gets3path(filepath)
    
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder): os.makedirs(folder)
    
    try:
      return s3.download_file(bucket, s3path, filepath)
    except ClientError as e:
      return e
    
    
  def upload(self,filepath):
    bucket = self.bucket
    s3path = self._gets3path(filepath)
    
    try:
      return s3.upload_file(filepath, bucket, s3path)
    except ClientError as e:
      return e
    