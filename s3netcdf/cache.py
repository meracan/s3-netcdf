import os
import shutil
import glob

    
class Cache(object):
  """
 
  
  Parameters
  ----------
 
  Attributes
  ----------

  
  """
  def __init__(self, parent):
    self.parent = parent
  
  def getNCs(self):
    folder = self.parent.folder
    return glob.glob(os.path.join(folder,"**/*.nc")) 
  

  def uploadNCA(self):
    self.parent.s3.upload(self.parent.ncaPath)
  
  def uploadNC(self):
    files = self.getNCs()
    for file in files:
      self.parent.s3.upload(file)
    
  def clearNCs(self):
    files = self.getNCs()
    for file in files:
      os.remove(file)
  
  def delete(self):
    shutil.rmtree(self.parent.folder)
    
  def clearOldest(self):
    if self.parent.localOnly:return
    cacheSize = self.parent.cacheSize
    files = self.getNCs()
    _files = [{"path":file,"stat":os.stat(file)} for file in files]
    _files.sort(key=lambda x: x['stat'].st_mtime, reverse=True)
    tsize=0
    for file in _files:
      fsize = file['stat'].st_size
      tsize +=fsize
      if(tsize>cacheSize):
        os.remove(file['path'])