import numpy as np

def getModelDatetime(startDate,endDate,step,ntime,extraDay=False):
  _ntime     = int((endDate-startDate)/step)+1
  if _ntime!=ntime:raise Exception("Check ntime")
  
  ndtime    = int(np.ceil((endDate-startDate)/np.timedelta64(1,"D")))
  ndtime    = ndtime+1 if extraDay else ndtime
  startYear = np.datetime64(startDate).astype('datetime64[Y]').astype(int) + 1970
  endYear   = np.datetime64(endDate).astype('datetime64[Y]').astype(int) + 1970
  nytime    = endYear-startYear+1
  sdecade   = int(np.floor(startYear/10.0)*10)
  edecade   = int(np.floor(endYear/10.0)*10)
  drange    = np.arange(sdecade,edecade+10,10)
  nDtime    = len(drange)
  stepUnits,step = np.datetime_data(step)
  
  return startDate,endDate,step,stepUnits,ntime,ndtime,nytime,nDtime


def createS3ModelDatetime(model):
  metadata  = model.obj['metadata']
  startdate = metadata['startDate']
  step      = metadata['step']
  stepUnit  = metadata['stepUnit']
  
  ntime     = model.dimensions['ntime']
  datetime=np.datetime64(startdate)+np.arange(ntime)*np.timedelta64(step, stepUnit)
  model['time','time']=datetime.astype("datetime64[s]")
  print(model['time','time'])
  
  if 'ndtime' in model.dimensions:
    ntime=model.dimensions['ndtime']
    datetime=np.datetime64(startdate)+np.arange(ntime)*np.timedelta64(24, 'h')
    model['dtime','dtime']=datetime.astype("datetime64[s]")
    print(model['dtime','dtime'])

  if 'nytime' in model.dimensions:
    nytime=model.dimensions['nytime']
    start=np.datetime64(startdate).astype('datetime64[Y]').astype(int) + 1970
    datetime=np.array([np.datetime64("{}-06-15T00:00".format(year)) for year in np.arange(start,start+nytime)])
    model['ytime','ytime']=datetime.astype("datetime64[s]")   
    
  if 'nDtime' in model.dimensions:
    nDtime=model.dimensions['nDtime']
    start=np.datetime64(startdate).astype('datetime64[Y]').astype(int) + 1970
    sdecade=int(np.floor((np.datetime64(startdate).astype("datetime64[Y]").astype(int)+1970)/10.0)*10+5)
    datetime=np.array([np.datetime64("{:n}-06-15T00:00".format(sdecade+decade*10)) for decade in np.arange(nDtime)])
    model['Dtime','Dtime']=datetime.astype("datetime64[s]")
    