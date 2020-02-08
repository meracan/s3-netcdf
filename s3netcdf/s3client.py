import os
import boto3
from botocore.errorfactory import ClientError

s3 = boto3.client('s3')

    
class S3Client(object):
  """
 
  
  Parameters
  ----------
 
  Attributes
  ----------

  
  """
  def __init__(self, parent):
    self.parent = parent
    self.bucket = parent.bucket
  
  def _gets3path(self,filepath):
    s3path = os.path.relpath(filepath,self.parent.cacheLocation)
    return s3path
    
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
    