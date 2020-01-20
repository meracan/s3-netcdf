

def main():
  ntime = 11
  nnode = 23
  
  data = np.arange(ntime * nnode) * -1.0
  data = np.reshape(data, (ntime, nnode))
  dataShape = data.shape
  masterShape = getMasterShape(data)
  
  filedata = np.resize(data, masterShape)
  
  # Looking
  index = np.ravel_multi_index(([1], [0]), dataShape)
  print(index)
  # print(np.ravel_multi_index([np.zeros(1000000,dtype=np.int)+1,np.arange(1000000,dtype=np.int)],[24*365,1000000]) )
  ii = np.unravel_index(index, masterShape)
  print(data[1, 0], filedata[ii])
