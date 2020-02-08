import os
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
  
  def _getFiles(self):
    folder = self.parent.folder
    return glob.glob(os.path.join(folder,"**/*.nc")) 
  
  def uploadNCA(self):
    self.parent.s3.upload(self.parent.ncaPath)
  
  def uploadNC(self):
    files = self._getFiles()
    for file in files:
      self.parent.s3.upload(file)
    
  def clear(self):
    files = self._getFiles()
    for file in files:
      os.remove(file)
  
  def clearOldest(self):
    if self.parent.localOnly:return
    cacheSize = self.parent.cacheSize
    files = self._getFiles()
    _files = [{"path":file,"stat":os.stat(file)} for file in files]
    _files.sort(key=lambda x: x['stat'].st_mtime, reverse=True)
    
    tsize=0
    for file in _files:
      fsize = file['stat'].st_size
      tsize +=fsize
      if(tsize>cacheSize):
        os.remove(file['path'])